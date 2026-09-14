"""
Orchestrator: wires the Exness client, live tick + event streams, the
Kelly sizer, the risk manager, and both signal modules into one running
loop.

WHAT THIS IS: a structurally complete skeleton — every piece connects to
every other piece correctly, and the parts that don't need a live account
(BarBuilder) are unit-tested. It will NOT run safely against a live
account as-is. See the checklist at the bottom before you touch real
money.

WHAT THIS DOES:
  1. Connects to Exness, confirms account access
  2. Streams live ticks, aggregating them into 1-minute bars per symbol
  3. Streams account/transaction events, so closed positions properly
     update the risk manager
  4. Every DECISION_INTERVAL_SECONDS, for each configured instrument with
     enough bar history and no open position: runs the matching signal
     (technical for forex, sentiment for crypto), gates it through the
     risk manager, sizes it with Kelly, and — if everything says go —
     submits the order with server-side stop-loss/take-profit attached.
     Exness's own servers then manage the exit; this script does not
     poll and manually close positions.

WHY NO HISTORICAL WARM-UP: exness_client.py doesn't have a verified
"get candle history" method — I could not confirm that endpoint's exact
path against the official docs (only its existence, by name, in a
navigation menu). So bar history is built from scratch out of live
ticks instead of fabricated. Practically: the forex signal needs 10
one-minute bars before it can fire, so expect ~10 minutes of silence
after every cold start. Wiring in real historical candles would remove
this wait, once you've confirmed that endpoint yourself.

BEFORE YOU RUN THIS FOR REAL:
  - Confirm open_position()'s request body schema against your own
    account (see exness_client.py's docstring) — this trusts it as-is
  - Confirm the field names in extract_close_pnl() (events_stream.py)
    and the tick payload fields used in _consume_ticks() below, against
    real messages from your account
  - Confirm how `volume` should actually be expressed for your account
    (lots vs. units) via get_instrument_condition() — _submit_order()
    below passes the risk fraction directly, which is almost certainly
    NOT the right format and is flagged inline
  - _open_fractions() is a placeholder — it doesn't yet track each open
    position's real size, only which symbols are open, so the
    correlation-cap and total-exposure numbers fed to the sizer/risk
    manager are not accurate yet
  - Start on a demo account with small size, not real capital, until
    everything above is confirmed
  - Both signals are templates, not proven edges (see their own
    docstrings) — this script trusts whatever they output

Requires: pip install requests cryptography websockets
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from backtest import Bar, SignalCall
from config import ACCOUNT_ID, API_HOST, API_KEY, PRIVATE_KEY_BYTES, WS_HOST
from events_stream import ExnessEventsStream, extract_close_pnl
from exness_client import ExnessClient
from kelly_position_sizer import KellyPositionSizer, OpenPosition
from risk_manager import RiskManager
from sentiment_signal_crypto import get_recent_sentiment, sentiment_to_signal
from technical_signal_forex import mean_reversion_signal
from ticks_stream import ExnessTicksStream

DECISION_INTERVAL_SECONDS = 60
BAR_INTERVAL_SECONDS = 60

# symbol -> "forex" or "crypto". Extend with whatever your account's real
# instrument list (client.get_instruments()) actually shows.
INSTRUMENTS = {
    "EURUSD": "forex",
    "GBPUSD": "forex",
    "BTCUSD": "crypto",
    "ETHUSD": "crypto",
}

# Only relevant for crypto instruments — LunarCrush topic name per symbol.
CRYPTO_TOPICS = {"BTCUSD": "bitcoin", "ETHUSD": "ethereum"}
LUNARCRUSH_API_KEY = "your-lunarcrush-key-here"  # move this into config.py in practice


@dataclass
class BarBuilder:
    """Buckets live ticks into fixed-interval OHLC bars, per symbol. Pure
    logic, no network — this is the part that's actually unit-tested."""

    interval_seconds: int
    bars: Dict[str, List[Bar]] = field(default_factory=dict)
    _current: Dict[str, dict] = field(default_factory=dict)

    def add_tick(self, symbol: str, price: float, ts: float) -> None:
        bucket = int(ts // self.interval_seconds) * self.interval_seconds
        cur = self._current.get(symbol)

        if cur is None or cur["bucket"] != bucket:
            if cur is not None:
                self.bars.setdefault(symbol, []).append(
                    Bar(
                        timestamp=str(cur["bucket"]),
                        open=cur["open"],
                        high=cur["high"],
                        low=cur["low"],
                        close=cur["close"],
                    )
                )
            self._current[symbol] = {"bucket": bucket, "open": price, "high": price, "low": price, "close": price}
        else:
            cur["high"] = max(cur["high"], price)
            cur["low"] = min(cur["low"], price)
            cur["close"] = price

    def get_bars(self, symbol: str) -> List[Bar]:
        return self.bars.get(symbol, [])


class TradingSystem:
    def __init__(self):
        self.client = ExnessClient(API_KEY, PRIVATE_KEY_BYTES, ACCOUNT_ID, API_HOST)
        account = self.client.get_account_info()
        # UNVERIFIED field name — adjust "balance" if your real account payload differs.
        balance = float(account.get("balance", 0))
        print(f"Connected. Account balance: {balance}")

        self.sizer = KellyPositionSizer(account_balance=balance)
        self.risk = RiskManager(state_path="risk_state.json", starting_equity=balance)
        self.bar_builder = BarBuilder(interval_seconds=BAR_INTERVAL_SECONDS)
        self.open_symbols: Set[str] = set()
        self.latest_price: Dict[str, float] = {}

        self.ticks = ExnessTicksStream(self.client, WS_HOST)
        self.events = ExnessEventsStream(self.client, WS_HOST)

    async def run(self):
        await asyncio.gather(
            self._consume_ticks(),
            self._consume_events(),
            self._decision_loop(),
        )

    async def _consume_ticks(self):
        async for tick in self.ticks.stream_ticks(["all"]):
            # UNVERIFIED field names — adjust to match real tick payloads.
            symbol = tick.get("instrument") or tick.get("symbol")
            price = tick.get("bid") or tick.get("price")
            if symbol and price:
                price = float(price)
                self.latest_price[symbol] = price
                self.bar_builder.add_tick(symbol, price, time.time())

    async def _consume_events(self):
        async for event in self.events.stream_events():
            symbol = event.get("instrument") or event.get("symbol")
            pnl = extract_close_pnl(event)
            if symbol and pnl is not None:
                print(f"Position closed: {symbol} pnl={pnl:+.2f}")
                self.risk.record_trade_result(pnl)
                self.open_symbols.discard(symbol)

    async def _decision_loop(self):
        while True:
            await asyncio.sleep(DECISION_INTERVAL_SECONDS)
            for symbol, asset_class in INSTRUMENTS.items():
                if symbol in self.open_symbols:
                    continue

                signal = self._get_signal(symbol, asset_class)
                if signal is None:
                    continue

                decision = self.risk.check_before_trade(
                    open_positions_count=len(self.open_symbols),
                    total_exposure_fraction=self._current_exposure(),
                )
                if not decision.allowed:
                    print(f"Blocked ({symbol}): {decision.reason}")
                    continue

                open_positions = [OpenPosition(s, f) for s, f in self._open_fractions().items()]
                sizing = self.sizer.size(
                    symbol, signal.win_probability, signal.reward_risk_ratio, open_positions=open_positions
                )
                if sizing.capped_fraction <= 0:
                    continue

                self._submit_order(symbol, signal, sizing.capped_fraction)

    def _get_signal(self, symbol: str, asset_class: str) -> Optional[SignalCall]:
        if asset_class == "forex":
            bars = self.bar_builder.get_bars(symbol)
            if len(bars) < 11:
                return None
            return mean_reversion_signal(bars)

        if asset_class == "crypto":
            price = self.latest_price.get(symbol)
            topic = CRYPTO_TOPICS.get(symbol)
            if price is None or topic is None:
                return None
            reading = get_recent_sentiment(topic, LUNARCRUSH_API_KEY)
            return sentiment_to_signal(reading, current_price=price)

        return None

    def _current_exposure(self) -> float:
        return sum(self._open_fractions().values())

    def _open_fractions(self) -> Dict[str, float]:
        # PLACEHOLDER: doesn't yet track each open position's real size,
        # only which symbols are open — so correlation-cap and
        # total-exposure numbers aren't accurate until this reads real
        # fill sizes back from the account/events stream.
        return {s: 0.0 for s in self.open_symbols}

    def _submit_order(self, symbol: str, signal: SignalCall, risk_fraction: float) -> None:
        price = self.latest_price.get(symbol)
        if price is None:
            return

        if signal.direction == "buy":
            stop_price = price - signal.stop_distance
            target_price = price + signal.stop_distance * signal.reward_risk_ratio
        else:
            stop_price = price + signal.stop_distance
            target_price = price - signal.stop_distance * signal.reward_risk_ratio

        # UNVERIFIED: how `volume` should actually be expressed for your
        # account (lots vs. units) — check get_instrument_condition()
        # before trusting this. Passing the raw risk fraction here is
        # almost certainly wrong; it's a placeholder, not a real sizing.
        volume = str(round(risk_fraction, 4))

        try:
            ack = self.client.open_position(
                instrument=symbol,
                side=signal.direction,
                volume=volume,
                stop_loss_price=str(stop_price),
                take_profit_price=str(target_price),
            )
            print(f"Order sent: {symbol} {signal.direction} vol={volume} -> {ack}")
            self.open_symbols.add(symbol)
        except Exception as e:
            print(f"Order failed for {symbol}: {e}")


if __name__ == "__main__":
    system = TradingSystem()
    asyncio.run(system.run())
