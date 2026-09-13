import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        fd, self.state_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.state_path)  # RiskManager should create it fresh

    def tearDown(self):
        if os.path.exists(self.state_path):
            os.remove(self.state_path)

    def _risk(self, **overrides):
        params = dict(
            state_path=self.state_path,
            starting_equity=1000.0,
            daily_loss_limit_pct=0.05,
            max_drawdown_from_peak_pct=0.15,
            max_consecutive_losses=4,
            max_open_positions=5,
            max_total_exposure_pct=0.20,
            on_breach=lambda msg: None,  # silence prints during tests
        )
        params.update(overrides)
        return RiskManager(**params)

    def test_allows_trade_under_all_limits(self):
        risk = self._risk()
        decision = risk.check_before_trade(open_positions_count=1, total_exposure_fraction=0.06)
        self.assertTrue(decision.allowed)

    def test_daily_loss_limit_halts_trading(self):
        risk = self._risk()
        risk.record_trade_result(pnl=-60.0)  # 6% of 1000, over the 5% limit
        decision = risk.check_before_trade(open_positions_count=0, total_exposure_fraction=0.0)
        self.assertFalse(decision.allowed)
        self.assertEqual(risk.status()["halt_reason"], "daily_loss_limit")

    def test_consecutive_losses_halts_trading_without_breaching_daily_limit(self):
        risk = self._risk()
        for _ in range(4):
            risk.record_trade_result(pnl=-5.0)  # 2% total, well under the 5% daily limit
        self.assertEqual(risk.status()["halt_reason"], "consecutive_losses")
        self.assertFalse(risk.check_before_trade(0, 0.0).allowed)

    def test_a_win_resets_consecutive_loss_counter(self):
        risk = self._risk()
        risk.record_trade_result(pnl=-5.0)
        risk.record_trade_result(pnl=-5.0)
        risk.record_trade_result(pnl=10.0)  # a win breaks the streak
        self.assertEqual(risk.status()["consecutive_losses"], 0)

    def test_drawdown_from_peak_halts_trading(self):
        risk = self._risk()
        risk.record_trade_result(pnl=200.0)  # equity 1200, new peak
        risk.record_trade_result(pnl=-190.0)  # equity 1010: -15.8% off the 1200 peak
        self.assertEqual(risk.status()["halt_reason"], "max_drawdown")

    def test_max_open_positions_blocks_even_when_not_halted(self):
        risk = self._risk()
        decision = risk.check_before_trade(open_positions_count=5, total_exposure_fraction=0.05)
        self.assertFalse(decision.allowed)

    def test_max_total_exposure_blocks_even_when_not_halted(self):
        risk = self._risk()
        decision = risk.check_before_trade(open_positions_count=1, total_exposure_fraction=0.25)
        self.assertFalse(decision.allowed)

    def test_manual_resume_clears_a_halt(self):
        risk = self._risk()
        risk.manual_halt("testing")
        self.assertFalse(risk.check_before_trade(0, 0.0).allowed)
        risk.manual_resume()
        self.assertTrue(risk.check_before_trade(0, 0.0).allowed)

    def test_state_persists_across_new_instances(self):
        risk1 = self._risk()
        risk1.record_trade_result(pnl=-30.0)
        risk2 = self._risk()  # simulates a process restart, same state_path
        self.assertEqual(risk2.status()["current_equity"], 970.0)


if __name__ == "__main__":
    unittest.main()
