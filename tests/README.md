# Kelly Position Sizer

Turns a (win probability, reward:risk ratio) estimate into a trade size,
with three layers of protection on top of raw Kelly:

1. **Fractional Kelly** — trades a fraction (default: half) of what full
   Kelly suggests.
2. **Hard per-trade cap** — never exceeds a fixed % of capital on one
   trade. Default 6%.
3. **Correlation-group cap** — caps total risk within a correlation
   group (e.g. EURUSD/GBPUSD/AUDUSD), not just per symbol.

No dependencies — standard library only.

## What this does NOT do

Estimate `win_probability` for you. That has to come from a real,
backtested model.

## Usage

python
from kelly_position_sizer import KellyPositionSizer, OpenPosition

sizer = KellyPositionSizer(account_balance=1000)
result = sizer.size("EURUSD", win_probability=0.58, reward_risk_ratio=1.5)
print(result)


## Tests

python3 -m unittest discover -s tests -v


## Disclaimer

This is a sizing tool, not trading advice. Forex/CFD trading is
high-risk — backtest any signal before it touches real capital.
