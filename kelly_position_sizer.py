"""
Kelly Position Sizer
=====================
Turns a (win probability, reward:risk ratio) estimate into a position size,
with three layers of protection on top of raw Kelly:

  1. Fractional Kelly   — trade a fraction (default: half) of what full
                           Kelly suggests. Full Kelly is provably growth-optimal
                           *only* if your probability estimate is exactly
                           right, and produces brutal drawdowns even then.
  2. Hard per-trade cap — never exceed a fixed % of capital on one trade,
                           regardless of what Kelly says. Default 6%.
  3. Correlation cap    — several "different" symbols can be one big bet in
                           disguise (EURUSD, GBPUSD, AUDUSD all move together
                           on broad USD strength). This caps total risk
                           within a correlation group, not just per symbol.

This module does NOT estimate win probability for you. That has to come
from a real model (technicals, fundamentals, sentiment, whatever) that's
been backtested — feeding it a guess produces a confident, precisely
wrong position size. Garbage in, Kelly-sized garbage out.

Usage:
    sizer = KellyPositionSizer(account_balance=1000)
    result = sizer.size("EURUSD", win_probability=0.58, reward_risk_ratio=1.5)
    print(result.position_size)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


# Rough default correlation groups for common Exness instruments.
# Extend/replace this for your own instrument list — it's a simplification,
# not a live correlation matrix.
DEFAULT_CORRELATION_GROUPS: Dict[str, str] = {
    "EURUSD": "usd_majors", "GBPUSD": "usd_majors", "AUDUSD": "usd_majors",
    "NZDUSD": "usd_majors", "USDJPY": "usd_majors", "USDCHF": "usd_majors",
    "USDCAD": "usd_majors",
    "EURGBP": "eur_crosses", "EURJPY": "eur_crosses", "EURCHF": "eur_crosses",
    "EURAUD": "eur_crosses", "EURCAD": "eur_crosses",
    "GBPJPY": "jpy_crosses", "AUDJPY": "jpy_crosses", "CADJPY": "jpy_crosses",
    "CHFJPY": "jpy_crosses", "NZDJPY": "jpy_crosses",
    "XAUUSD": "metals", "XAGUSD": "metals",
    "BTCUSD": "crypto", "ETHUSD": "crypto", "XRPUSD": "crypto",
    "US30": "indices", "US500": "indices", "NAS100": "indices",
    "UK100": "indices", "GER40": "indices",
}


@dataclass
class OpenPosition:
    symbol: str
    risk_fraction: float  # fraction of account currently at risk on this position


@dataclass
class SizingResult:
    symbol: str
    win_probability_used: float
    reward_risk_ratio: float
    raw_kelly_fraction: float
    fractional_kelly_fraction: float
    capped_fraction: float
    position_size: float
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"{self.symbol}: size = {self.position_size:,.2f} "
            f"({self.capped_fraction:.2%} of account)",
            f"  raw Kelly: {self.raw_kelly_fraction:.2%}  "
            f"-> fractional Kelly: {self.fractional_kelly_fraction:.2%}  "
            f"-> capped: {self.capped_fraction:.2%}",
        ]
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


class KellyPositionSizer:
    def __init__(
        self,
        account_balance: float,
        kelly_fraction: float = 0.5,             # half-Kelly by default
        hard_cap_per_trade: float = 0.06,         # 6% ceiling, non-negotiable
        max_correlated_exposure: float = 0.12,    # cap per correlation group
        max_probability: float = 0.75,            # distrust p above this
        correlation_groups: Optional[Dict[str, str]] = None,
    ):
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        if not 0 < hard_cap_per_trade <= 1:
            raise ValueError("hard_cap_per_trade must be in (0, 1]")

        self.account_balance = account_balance
        self.kelly_fraction = kelly_fraction
        self.hard_cap_per_trade = hard_cap_per_trade
        self.max_correlated_exposure = max_correlated_exposure
        self.max_probability = max_probability
        self.correlation_groups = correlation_groups or DEFAULT_CORRELATION_GROUPS

    def _group_for(self, symbol: str) -> str:
        return self.correlation_groups.get(symbol.upper(), symbol.upper())

    def _group_exposure(self, symbol: str, open_positions: List[OpenPosition]) -> float:
        group = self._group_for(symbol)
        return sum(
            p.risk_fraction for p in open_positions if self._group_for(p.symbol) == group
        )

    def size(
        self,
        symbol: str,
        win_probability: float,
        reward_risk_ratio: float,
        open_positions: Optional[List[OpenPosition]] = None,
    ) -> SizingResult:
        """
        win_probability: your estimated P(win) for this trade, e.g. 0.58
        reward_risk_ratio: reward-to-risk, e.g. 1.5 means "risk 1 to make 1.5"
        open_positions: currently open trades, for correlation-cap checking
        """
        open_positions = open_positions or []
        warnings: List[str] = []
        p, b = win_probability, reward_risk_ratio

        if not 0 < p < 1:
            raise ValueError("win_probability must be between 0 and 1")
        if b <= 0:
            raise ValueError("reward_risk_ratio must be positive")

        if p > self.max_probability:
            warnings.append(
                f"win_probability {p:.0%} exceeds the {self.max_probability:.0%} sanity "
                f"ceiling and has been clamped. Overconfident probability estimates — "
                f"especially from noisy signals like social sentiment — are the most "
                f"common way Kelly-sized accounts blow up."
            )
            p = self.max_probability

        raw_kelly = p - (1 - p) / b
        if raw_kelly <= 0:
            warnings.append(
                f"No edge at p={p:.0%}, reward:risk={b:.2f} (raw Kelly <= 0). Sizing to 0."
            )
            raw_kelly = frac_kelly = capped = 0.0
        else:
            frac_kelly = raw_kelly * self.kelly_fraction
            capped = min(frac_kelly, self.hard_cap_per_trade)
            if frac_kelly > self.hard_cap_per_trade:
                warnings.append(
                    f"{self.kelly_fraction:.0%}-Kelly suggested {frac_kelly:.2%} of capital — "
                    f"capped at the {self.hard_cap_per_trade:.0%} per-trade ceiling."
                )

            group = self._group_for(symbol)
            existing = self._group_exposure(symbol, open_positions)
            if existing + capped > self.max_correlated_exposure:
                allowed = max(0.0, self.max_correlated_exposure - existing)
                same_group = [p_.symbol for p_ in open_positions if self._group_for(p_.symbol) == group]
                warnings.append(
                    f"'{group}' group already has {existing:.2%} at risk ({same_group}). "
                    f"Reducing size from {capped:.2%} to {allowed:.2%} to respect the "
                    f"{self.max_correlated_exposure:.0%} group cap."
                )
                capped = allowed

        return SizingResult(
            symbol=symbol,
            win_probability_used=p,
            reward_risk_ratio=b,
            raw_kelly_fraction=raw_kelly,
            fractional_kelly_fraction=frac_kelly,
            capped_fraction=capped,
            position_size=self.account_balance * capped,
            warnings=warnings,
        )


if __name__ == "__main__":
    sizer = KellyPositionSizer(account_balance=1000)

    print("--- Scenario 1: modest, believable edge ---")
    print(sizer.size("EURUSD", win_probability=0.58, reward_risk_ratio=1.5))

    print("\n--- Scenario 2: overconfident sentiment-driven estimate ---")
    print(sizer.size("BTCUSD", win_probability=0.85, reward_risk_ratio=1.2))

    print("\n--- Scenario 3: correlated exposure already open ---")
    open_positions = [OpenPosition(symbol="EURUSD", risk_fraction=0.08)]
    print(sizer.size("GBPUSD", win_probability=0.60, reward_risk_ratio=1.4,
                      open_positions=open_positions))

    print("\n--- Scenario 4: no real edge ---")
    print(sizer.size("USDJPY", win_probability=0.51, reward_risk_ratio=0.9))
