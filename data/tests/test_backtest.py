import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backtest import Bar, BacktestEngine, SignalCall
from kelly_position_sizer import KellyPositionSizer
from risk_manager import RiskManager


def make_bars(closes):
    """Simple bars where high/low are +/-0.001 around the close, for easy control."""
    return [
        Bar(timestamp=f"day{i}", open=c, high=c + 0.001, low=c - 0.001, close=c)
        for i, c in enumerate(closes)
    ]


class TestBacktestEngine(unittest.TestCase):
    def setUp(self):
        self.sizer = KellyPositionSizer(account_balance=1000.0, hard_cap_per_trade=0.06)

    def test_no_signal_means_no_trades(self):
        bars = make_bars([1.10] * 30)
        engine = BacktestEngine(bars, self.sizer, signal_fn=lambda b: None, warmup_bars=5)
        result = engine.run()
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.final_equity, result.starting_equity)

    def test_buy_hits_take_profit(self):
        # Flat, then a signal fires once, then price climbs straight to the target.
        closes = [1.10] * 10 + [1.10, 1.11, 1.12, 1.13, 1.14, 1.15]
        bars = make_bars(closes)

        def signal_fn(bars_so_far):
            if len(bars_so_far) == 11:  # fire exactly once, right after warmup
                return SignalCall(direction="buy", win_probability=0.6, reward_risk_ratio=1.5, stop_distance=0.02)
            return None

        engine = BacktestEngine(bars, self.sizer, signal_fn=signal_fn, warmup_bars=10, max_holding_bars=10)
        result = engine.run()

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.outcome, "take_profit")
        self.assertGreater(trade.pnl, 0)
        # capped at 6%, reward:risk 1.5 -> pnl should be ~6% * 1.5 * starting equity
        self.assertAlmostEqual(trade.pnl, 1000.0 * 0.06 * 1.5, places=0)

    def test_buy_hits_stop_loss(self):
        closes = [1.10] * 10 + [1.10, 1.09, 1.08, 1.07, 1.06, 1.05]
        bars = make_bars(closes)

        def signal_fn(bars_so_far):
            if len(bars_so_far) == 11:
                return SignalCall(direction="buy", win_probability=0.6, reward_risk_ratio=1.5, stop_distance=0.02)
            return None

        engine = BacktestEngine(bars, self.sizer, signal_fn=signal_fn, warmup_bars=10, max_holding_bars=10)
        result = engine.run()

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.outcome, "stop_loss")
        self.assertLess(trade.pnl, 0)
        self.assertAlmostEqual(trade.pnl, -1000.0 * 0.06, places=0)

    def test_timeout_when_neither_level_is_hit(self):
        closes = [1.10] * 10 + [1.10, 1.101, 1.099, 1.1005, 1.0995, 1.1002]
        bars = make_bars(closes)

        def signal_fn(bars_so_far):
            if len(bars_so_far) == 11:
                return SignalCall(direction="buy", win_probability=0.6, reward_risk_ratio=1.5, stop_distance=0.05)
            return None

        engine = BacktestEngine(bars, self.sizer, signal_fn=signal_fn, warmup_bars=10, max_holding_bars=5)
        result = engine.run()

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].outcome, "timeout")

    def test_zero_probability_edge_produces_no_trade(self):
        bars = make_bars([1.10] * 20)

        def signal_fn(bars_so_far):
            if len(bars_so_far) == 11:
                return SignalCall(direction="buy", win_probability=0.50, reward_risk_ratio=1.0, stop_distance=0.02)
            return None

        engine = BacktestEngine(bars, self.sizer, signal_fn=signal_fn, warmup_bars=10)
        result = engine.run()
        self.assertEqual(len(result.trades), 0)  # no edge -> sizer returns 0 -> no trade

    def test_halted_risk_manager_blocks_all_trades(self):
        bars = make_bars([1.10] * 10 + [1.10, 1.11, 1.12, 1.13, 1.14, 1.15])

        def signal_fn(bars_so_far):
            if len(bars_so_far) == 11:
                return SignalCall(direction="buy", win_probability=0.6, reward_risk_ratio=1.5, stop_distance=0.02)
            return None

        import tempfile
        fd, state_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(state_path)
        risk = RiskManager(state_path=state_path, starting_equity=1000.0, on_breach=lambda m: None)
        risk.manual_halt("testing")

        engine = BacktestEngine(bars, self.sizer, signal_fn=signal_fn, warmup_bars=10, risk=risk)
        result = engine.run()

        self.assertEqual(len(result.trades), 0)
        os.remove(state_path)

    def test_max_drawdown_calculation(self):
        bars = make_bars([1.10] * 30)
        engine = BacktestEngine(bars, self.sizer, signal_fn=lambda b: None, warmup_bars=5, starting_equity=1000.0)
        result = engine.run()
        result.equity_curve = [1000, 1100, 900, 950]  # manually set to test the math
        self.assertAlmostEqual(result.max_drawdown_pct, (1 - 900 / 1100) * 100, places=4)


if __name__ == "__main__":
    unittest.main()
