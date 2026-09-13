import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kelly_position_sizer import KellyPositionSizer, OpenPosition


class TestKellyPositionSizer(unittest.TestCase):
    def setUp(self):
        self.sizer = KellyPositionSizer(account_balance=1000)

    def test_no_edge_at_50_50(self):
        result = self.sizer.size("EURUSD", win_probability=0.50, reward_risk_ratio=1.0)
        self.assertEqual(result.capped_fraction, 0.0)
        self.assertEqual(result.position_size, 0.0)

    def test_negative_edge_sizes_to_zero(self):
        result = self.sizer.size("USDJPY", win_probability=0.51, reward_risk_ratio=0.9)
        self.assertEqual(result.capped_fraction, 0.0)
        self.assertTrue(any("No edge" in w for w in result.warnings))

    def test_hard_cap_engages_on_strong_edge(self):
        result = self.sizer.size("EURUSD", win_probability=0.58, reward_risk_ratio=1.5)
        self.assertAlmostEqual(result.capped_fraction, 0.06, places=6)
        self.assertTrue(any("capped" in w for w in result.warnings))

    def test_probability_sanity_clamp(self):
        result = self.sizer.size("BTCUSD", win_probability=0.95, reward_risk_ratio=1.2)
        self.assertLessEqual(result.win_probability_used, 0.75)
        self.assertTrue(any("sanity ceiling" in w for w in result.warnings))

    def test_correlation_cap_reduces_size(self):
        open_positions = [OpenPosition(symbol="EURUSD", risk_fraction=0.08)]
        result = self.sizer.size(
            "GBPUSD", win_probability=0.60, reward_risk_ratio=1.4,
            open_positions=open_positions,
        )
        self.assertAlmostEqual(result.capped_fraction, 0.04, places=6)
        self.assertTrue(any("group cap" in w for w in result.warnings))

    def test_uncorrelated_symbol_not_affected_by_other_groups(self):
        open_positions = [OpenPosition(symbol="EURUSD", risk_fraction=0.08)]
        result = self.sizer.size(
            "XAUUSD", win_probability=0.58, reward_risk_ratio=1.5,
            open_positions=open_positions,
        )
        self.assertAlmostEqual(result.capped_fraction, 0.06, places=6)

    def test_invalid_probability_raises(self):
        with self.assertRaises(ValueError):
            self.sizer.size("EURUSD", win_probability=1.2, reward_risk_ratio=1.5)

    def test_invalid_reward_risk_raises(self):
        with self.assertRaises(ValueError):
            self.sizer.size("EURUSD", win_probability=0.6, reward_risk_ratio=0)


if __name__ == "__main__":
    unittest.main()
