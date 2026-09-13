"""
Risk manager / circuit breakers.

Sits between "the signal says trade X" and "actually send the order to
Exness." Every proposed trade — and every closed trade's result — passes
through here. This is what turns "the code technically can trade" into
"the code won't quietly destroy the account while nobody's watching."

Intended integration with the rest of this repo:
    1. risk.check_before_trade(...) — must return allowed=True before you
       even compute a size
    2. kelly_position_sizer.KellyPositionSizer.size(...) — decide how big
    3. exness_client.ExnessClient.open_position(...) — send it
    4. once the position closes (via the server events stream), call
       risk.record_trade_result(pnl) so the circuit breakers stay accurate

State (today's P&L, peak equity, consecutive losses, halt status) is
persisted to a small JSON file so a process restart doesn't wipe the
safety rails. No external dependencies — stdlib only.
"""

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional


def _default_on_breach(message: str) -> None:
    # Replace this with a real notification (Telegram, email, SMS — whatever
    # actually reaches you) once you're running this unattended. A halt that
    # only prints to a log nobody's reading defeats the point.
    print(f"[risk] {message}")


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(
        self,
        state_path: str,
        starting_equity: float,
        daily_loss_limit_pct: float = 0.05,
        max_drawdown_from_peak_pct: float = 0.15,
        max_consecutive_losses: int = 4,
        max_open_positions: int = 5,
        max_total_exposure_pct: float = 0.20,
        on_breach: Optional[Callable[[str], None]] = None,
    ):
        """
        daily_loss_limit_pct: pause new trades for the rest of the day once
            today's loss hits this fraction of the day's starting equity.
            Clears automatically at the next day's rollover.
        max_drawdown_from_peak_pct: full halt once equity falls this far
            below its all-time peak. Does NOT auto-clear — call
            manual_resume() once you've actually looked at what happened.
        max_consecutive_losses: full halt after this many losing trades in
            a row. Also does not auto-clear.
        max_open_positions / max_total_exposure_pct: portfolio-wide caps
            checked before every new trade, independent of the Kelly
            sizer's own per-trade and per-correlation-group caps.
        on_breach: callback fired with a message every time a halt (or
            resume) happens. Defaults to printing — wire this to a real
            alert channel for actual unattended use.
        """
        self.state_path = state_path
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_from_peak_pct = max_drawdown_from_peak_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_open_positions = max_open_positions
        self.max_total_exposure_pct = max_total_exposure_pct
        self.on_breach = on_breach or _default_on_breach

        self.state = self._load_state(starting_equity)
        self._roll_to_new_day_if_needed()

    # ---- persistence ----

    def _load_state(self, starting_equity: float) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                return json.load(f)
        return {
            "current_day": str(date.today()),
            "day_start_equity": starting_equity,
            "peak_equity": starting_equity,
            "current_equity": starting_equity,
            "consecutive_losses": 0,
            "halted": False,
            "halt_reason": None,
        }

    def _save_state(self) -> None:
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def _roll_to_new_day_if_needed(self) -> None:
        today = str(date.today())
        if self.state["current_day"] != today:
            self.state["current_day"] = today
            self.state["day_start_equity"] = self.state["current_equity"]
            # Only the daily-loss halt clears automatically at day rollover.
            # Drawdown and consecutive-loss halts need a human to look and
            # call manual_resume() on purpose.
            if self.state.get("halt_reason") == "daily_loss_limit":
                self.state["halted"] = False
                self.state["halt_reason"] = None
            self._save_state()

    # ---- recording outcomes ----

    def record_trade_result(self, pnl: float) -> None:
        """Call once a position closes, with its realized P&L (+/-, account currency)."""
        self._roll_to_new_day_if_needed()
        self.state["current_equity"] += pnl
        self.state["peak_equity"] = max(self.state["peak_equity"], self.state["current_equity"])
        self.state["consecutive_losses"] = self.state["consecutive_losses"] + 1 if pnl < 0 else 0

        self._check_and_apply_halts()
        self._save_state()

    def _check_and_apply_halts(self) -> None:
        equity = self.state["current_equity"]

        if self.state["day_start_equity"] > 0:
            day_loss_pct = 1 - (equity / self.state["day_start_equity"])
            if day_loss_pct >= self.daily_loss_limit_pct and not self.state["halted"]:
                self._halt(
                    "daily_loss_limit",
                    f"Daily loss {day_loss_pct:.2%} hit the {self.daily_loss_limit_pct:.0%} limit.",
                )

        if self.state["peak_equity"] > 0:
            drawdown_pct = 1 - (equity / self.state["peak_equity"])
            if drawdown_pct >= self.max_drawdown_from_peak_pct and not self.state["halted"]:
                self._halt(
                    "max_drawdown",
                    f"Drawdown from peak {drawdown_pct:.2%} hit the "
                    f"{self.max_drawdown_from_peak_pct:.0%} limit.",
                )

        if self.state["consecutive_losses"] >= self.max_consecutive_losses and not self.state["halted"]:
            self._halt(
                "consecutive_losses",
                f"{self.state['consecutive_losses']} losing trades in a row.",
            )

    def _halt(self, reason: str, message: str) -> None:
        self.state["halted"] = True
        self.state["halt_reason"] = reason
        self.on_breach(f"TRADING HALTED ({reason}): {message}")

    # ---- manual controls ----

    def manual_halt(self, message: str = "Manually halted.") -> None:
        self._halt("manual", message)
        self._save_state()

    def manual_resume(self) -> None:
        self.state["halted"] = False
        self.state["halt_reason"] = None
        self._save_state()
        self.on_breach("Trading manually resumed.")

    # ---- the gate every proposed trade must pass ----

    def check_before_trade(self, open_positions_count: int, total_exposure_fraction: float) -> RiskDecision:
        self._roll_to_new_day_if_needed()

        if self.state["halted"]:
            return RiskDecision(
                False, f"Trading halted ({self.state['halt_reason']}). Call manual_resume() once reviewed."
            )

        if open_positions_count >= self.max_open_positions:
            return RiskDecision(False, f"Already at max open positions ({self.max_open_positions}).")

        if total_exposure_fraction >= self.max_total_exposure_pct:
            return RiskDecision(
                False,
                f"Total exposure {total_exposure_fraction:.2%} already at the "
                f"{self.max_total_exposure_pct:.0%} portfolio cap.",
            )

        return RiskDecision(True, "OK")

    def status(self) -> dict:
        return dict(self.state)


if __name__ == "__main__":
    import tempfile

    state_file = os.path.join(tempfile.gettempdir(), "risk_manager_demo_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)

    risk = RiskManager(
        state_path=state_file,
        starting_equity=1000.0,
        daily_loss_limit_pct=0.05,
        max_drawdown_from_peak_pct=0.15,
        max_consecutive_losses=4,
        max_open_positions=5,
        max_total_exposure_pct=0.20,
    )

    print("--- Normal trade check ---")
    print(risk.check_before_trade(open_positions_count=1, total_exposure_fraction=0.06))

    print("\n--- A small loss ---")
    risk.record_trade_result(pnl=-5.0)
    print(risk.status())

    print("\n--- Four losses in a row trips the consecutive-loss breaker ---")
    print("    (kept small on purpose so this trips on ITS OWN, not the daily-loss limit too)")
    risk.record_trade_result(pnl=-5.0)
    risk.record_trade_result(pnl=-5.0)
    risk.record_trade_result(pnl=-5.0)
    print(risk.status())
    print(risk.check_before_trade(open_positions_count=0, total_exposure_fraction=0.0))

    print("\n--- Manually resuming after review ---")
    risk.manual_resume()
    print(risk.check_before_trade(open_positions_count=0, total_exposure_fraction=0.0))

    print("\n--- Too many open positions blocks a new trade even though not halted ---")
    print(risk.check_before_trade(open_positions_count=5, total_exposure_fraction=0.10))

    os.remove(state_file)
