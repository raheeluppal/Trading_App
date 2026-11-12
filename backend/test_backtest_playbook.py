"""Execution-path checks for the historical playbook simulator."""

import unittest

import pandas as pd

from backtest_playbook import _exit_price, backtest
from test_playbook import bullish_case


class BacktestTests(unittest.TestCase):
    def test_both_hit_candle_uses_stop(self):
        bar = pd.Series({"open": 100.0, "high": 105.0, "low": 95.0})
        self.assertEqual(_exit_price("BUY", bar, 98.0, 104.0), (98.0, "stop"))
        self.assertEqual(_exit_price("PUT", bar, 102.0, 96.0), (102.0, "stop"))

    def test_signal_needs_next_bar_to_fill(self):
        one, five, hour, day, _ = bullish_case()
        self.assertEqual(backtest(one, five, hour, day, sessions=1)["trades"], 0)
        next_time = one.index[-1] + pd.Timedelta(minutes=1)
        one.loc[next_time, ["open", "high", "low", "close", "volume"]] = [
            105.21, 105.3, 104.0, 104.7, 100,
        ]
        result = backtest(one, five, hour, day, sessions=1)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["trades_detail"][0]["reason"], "stop")


if __name__ == "__main__":
    unittest.main()
