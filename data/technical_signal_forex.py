"""
A starter technical signal for forex pairs: mean reversion off a rolling
average, sized by how far price has stretched from it.

THIS IS A TEMPLATE, NOT A VALIDATED EDGE. The win_probability mapping below
(distance-from-average -> made-up probability) is a reasonable-looking
guess, not something backed by research. The entire point of running it
through backtest.py is to find out — on real historical data — whether
this particular guess produces a positive expectancy, a negative one, or
nothing distinguishable from noise. Don't skip straight to trusting it.

Data: loads real EUR/USD daily OHLC pulled from Alpha Vantage
(data/eurusd_daily.json, ~100 trading days, Apr-Sep 2026).
"""

import json
import statistics
from typing import List, Optional

from backtest import Bar, BacktestEngine, SignalCall
from kelly_position_sizer import KellyPositionSizer


def load_eurusd_bars(path: str = "data/eurusd_daily.json") -> List[Bar]:
    with open(path, "r") as f:
        raw = json.load(f)

    series = raw["Time Series FX (Daily)"]
    bars = [
        Bar(
            timestamp=ts,
            open=float(v["1. open"]),
            high=float(v["2. high"]),
            low=float(v["3. low"]),
            close=float(v["4. close"]),
        )
        for ts, v in series.items()
    ]
    bars.sort(key=lambda b: b.timestamp)  # Alpha Vantage returns newest-first
    return bars


def mean_reversion_signal(
    bars_so_far: List[Bar],
    lookback: int = 10,
    z_threshold: float = 1.0,
    reward_risk_ratio: float = 1.5,
) -> Optional[SignalCall]:
    """
    If the latest close has stretched more than z_threshold standard
    deviations away from its `lookback`-bar average, bet on it reverting.
    Stop is placed at one average daily range away — a plain, explainable
    starting point, not an optimized one.
    """
    if len(bars_so_far) < lookback + 1:
        return None

    window = bars_so_far[-lookback:]
    closes = [b.close for b in window]
    mean_close = statistics.mean(closes)
    stdev_close = statistics.pstdev(closes)
    if stdev_close == 0:
        return None

    latest = bars_so_far[-1]
    z_score = (latest.close - mean_close) / stdev_close

    avg_range = statistics.mean(b.high - b.low for b in window)
    if avg_range <= 0:
        return None

    if z_score <= -z_threshold:
        direction = "buy"   # stretched below average -> bet on reverting up
    elif z_score >= z_threshold:
        direction = "sell"  # stretched above average -> bet on reverting down
    else:
        return None

    # Made-up mapping, exactly what the backtest is meant to interrogate:
    # bigger stretch -> assumed higher probability of reverting, capped.
    win_probability = min(0.70, 0.50 + abs(z_score) * 0.05)

    return SignalCall(
        direction=direction,
        win_probability=win_probability,
        reward_risk_ratio=reward_risk_ratio,
        stop_distance=avg_range,
    )


if __name__ == "__main__":
    bars = load_eurusd_bars()
    print(f"Loaded {len(bars)} real EURUSD daily bars: {bars[0].timestamp} to {bars[-1].timestamp}")

    sizer = KellyPositionSizer(account_balance=1000.0)
    engine = BacktestEngine(
        bars=bars,
        sizer=sizer,
        signal_fn=mean_reversion_signal,
        symbol="EURUSD",
        warmup_bars=10,
        max_holding_bars=10,
        starting_equity=1000.0,
        spread_price_units=0.00010,  # ~1 pip, a rough EURUSD spread stand-in
    )
    result = engine.run()

    print("\n--- Backtest result (real data, toy signal) ---")
    print(result.summary())

    print("\n--- Trade log ---")
    for t in result.trades:
        print(
            f"{t.entry_timestamp} {t.direction:4s} @ {t.entry_price:.5f} -> "
            f"{t.exit_timestamp} {t.outcome:11s} @ {t.exit_price:.5f}  pnl={t.pnl:+.2f}"
  )
