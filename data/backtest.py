"""
Generic backtest engine.

Walks forward through historical bars one at a time, calling your signal
function with ONLY the data up to that point (never the future — that's
the #1 way backtests lie to you). When a signal fires, sizes it with
KellyPositionSizer, optionally gates it through RiskManager, then
simulates the trade forward bar-by-bar until it hits its stop, its
target, or a max holding period.

What this does NOT do:
  - Model spread/commission by default (optional `spread_price_units` gets
    you partway there, but real transaction costs will eat into whatever
    edge this shows)
  - Handle more than one open position at a time (v1 — extend if you want
    portfolio-level backtesting)
  - Prove a signal is good. A profitable backtest on ~100 days of data is
    a hypothesis, not a result — small samples overfit easily. Treat any
    result here as "worth investigating further," not "ready to trade."

Requires: nothing beyond the standard library.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from kelly_position_sizer import KellyPositionSizer
from risk_manager import RiskManager


@dataclass
class Bar:
    timestamp: str
    open: float
    high: float
    low: float
    close: float


@dataclass
class SignalCall:
    direction: str          # "buy" or "sell"
    win_probability: float  # your model's estimated P(win), e.g. 0.58
    reward_risk_ratio: float  # e.g. 1.5 = risking 1 to make 1.5
    stop_distance: float    # in price units, always positive


@dataclass
class TradeResult:
    entry_index: int
    entry_timestamp: str
    entry_price: float
    exit_index: int
    exit_timestamp: str
    exit_price: float
    direction: str
    outcome: str  # "take_profit", "stop_loss", or "timeout"
    risk_fraction: float
    pnl: float


@dataclass
class BacktestResult:
    trades: List[TradeResult] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    starting_equity: float = 0.0
    final_equity: float = 0.0

    @property
    def total_return_pct(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.final_equity / self.starting_equity - 1) * 100

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        peak = self.equity_curve[0] if self.equity_curve else 0.0
        max_dd = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, 1 - equity / peak)
        return max_dd * 100

    def summary(self) -> str:
        lines = [
            f"Trades: {len(self.trades)}",
            f"Win rate: {self.win_rate:.1%}",
            f"Total return: {self.total_return_pct:+.2f}%",
            f"Max drawdown: {self.max_drawdown_pct:.2f}%",
            f"Final equity: {self.final_equity:,.2f} (from {self.starting_equity:,.2f})",
        ]
        return "\n".join(lines)


class BacktestEngine:
    def __init__(
        self,
        bars: List[Bar],
        sizer: KellyPositionSizer,
        signal_fn: Callable[[List[Bar]], Optional[SignalCall]],
        symbol: str = "TEST",
        warmup_bars: int = 20,
        max_holding_bars: int = 20,
        starting_equity: float = 1000.0,
        spread_price_units: float = 0.0,
        risk: Optional[RiskManager] = None,
    ):
        """
        bars: full historical series, oldest first
        signal_fn: called as signal_fn(bars[:i+1]) at each step — sees only
            data up to and including the current bar, never the future.
            Return None for "no trade," or a SignalCall to propose one.
        warmup_bars: how many bars to skip before the first possible signal
            (so signal_fn always has some history to look back on)
        max_holding_bars: force-exit at market if neither stop nor target
            is hit within this many bars
        spread_price_units: a fixed cost in price units subtracted at
            entry, as a crude stand-in for bid/ask spread. 0 = ignored,
            which overstates real-world results.
        risk: optional RiskManager. If given, every signal is gated through
            check_before_trade() and every close calls record_trade_result().
            NOTE: RiskManager's daily-loss limit is tied to the real
            wall-clock date, not the bar's historical date — since a
            backtest runs in seconds, that breaker effectively treats the
            whole run as a single day. Drawdown and consecutive-loss
            breakers aren't date-based and work correctly here.
        """
        self.bars = bars
        self.sizer = sizer
        self.signal_fn = signal_fn
        self.symbol = symbol
        self.warmup_bars = warmup_bars
        self.max_holding_bars = max_holding_bars
        self.starting_equity = starting_equity
        self.spread_price_units = spread_price_units
        self.risk = risk

    def run(self) -> BacktestResult:
        equity = self.starting_equity
        equity_curve = [equity]
        trades: List[TradeResult] = []
        open_pos = None
        i = self.warmup_bars

        while i < len(self.bars):
            bar = self.bars[i]

            if open_pos is None:
                signal = self.signal_fn(self.bars[: i + 1])
                if signal is not None and signal.stop_distance > 0:
                    allowed = True
                    if self.risk is not None:
                        decision = self.risk.check_before_trade(
                            open_positions_count=0, total_exposure_fraction=0.0
                        )
                        allowed = decision.allowed

                    if allowed:
                        sizing = self.sizer.size(
                            self.symbol, signal.win_probability, signal.reward_risk_ratio
                        )
                        if sizing.capped_fraction > 0:
                            entry_price = bar.close
                            if signal.direction == "buy":
                                entry_price += self.spread_price_units / 2
                                stop_price = entry_price - signal.stop_distance
                                target_price = entry_price + signal.stop_distance * signal.reward_risk_ratio
                            else:
                                entry_price -= self.spread_price_units / 2
                                stop_price = entry_price + signal.stop_distance
                                target_price = entry_price - signal.stop_distance * signal.reward_risk_ratio

                            open_pos = {
                                "entry_index": i,
                                "entry_price": entry_price,
                                "direction": signal.direction,
                                "stop_price": stop_price,
                                "target_price": target_price,
                                "stop_distance": signal.stop_distance,
                                "risk_fraction": sizing.capped_fraction,
                                "deadline_index": i + self.max_holding_bars,
                            }
            else:
                outcome, exit_price = self._check_exit(bar, open_pos, i)
                if outcome is not None:
                    if open_pos["direction"] == "buy":
                        raw_move = exit_price - open_pos["entry_price"]
                    else:
                        raw_move = open_pos["entry_price"] - exit_price

                    move_in_r = raw_move / open_pos["stop_distance"]
                    pnl = open_pos["risk_fraction"] * equity * move_in_r
                    equity += pnl

                    trades.append(
                        TradeResult(
                            entry_index=open_pos["entry_index"],
                            entry_timestamp=self.bars[open_pos["entry_index"]].timestamp,
                            entry_price=open_pos["entry_price"],
                            exit_index=i,
                            exit_timestamp=bar.timestamp,
                            exit_price=exit_price,
                            direction=open_pos["direction"],
                            outcome=outcome,
                            risk_fraction=open_pos["risk_fraction"],
                            pnl=pnl,
                        )
                    )
                    if self.risk is not None:
                        self.risk.record_trade_result(pnl)
                    open_pos = None

            equity_curve.append(equity)
            i += 1

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            starting_equity=self.starting_equity,
            final_equity=equity,
        )

    @staticmethod
    def _check_exit(bar: Bar, open_pos: dict, i: int):
        if open_pos["direction"] == "buy":
            if bar.low <= open_pos["stop_price"]:
                return "stop_loss", open_pos["stop_price"]
            if bar.high >= open_pos["target_price"]:
                return "take_profit", open_pos["target_price"]
        else:
            if bar.high >= open_pos["stop_price"]:
                return "stop_loss", open_pos["stop_price"]
            if bar.low <= open_pos["target_price"]:
                return "take_profit", open_pos["target_price"]

        if i >= open_pos["deadline_index"]:
            return "timeout", bar.close

        return None, None
