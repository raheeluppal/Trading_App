"""Execution-path checks for the historical playbook simulator."""

import unittest

import pandas as pd

from backtest_playbook import _exit_price, _stock_result, backtest
from test_playbook import bullish_case
from zone_history import validate_zone_history


class BacktestTests(unittest.TestCase):
    def test_stock_long_and_short_dollar_pnl(self):
        long = _stock_result("BUY", 100.0, 101.0, 8, 10)
        short = _stock_result("PUT", 100.0, 99.0, 8, 10)
        self.assertEqual(long["stock_side"], "LONG")
        self.assertEqual(short["stock_side"], "SHORT")
        self.assertEqual(long["net_pnl_dollars"], 9.2)
        self.assertEqual(short["net_pnl_dollars"], 9.2)
        self.assertEqual(long["net_return_pct"], 0.92)
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

    def test_open_trade_can_exit_after_11_am(self):
        one, five, hour, day, _ = bullish_case()
        times = pd.date_range(one.index[-1] + pd.Timedelta(minutes=1), periods=35, freq="min")
        continuation = pd.DataFrame(
            {"open": 105.2, "high": 105.3, "low": 105.1, "close": 105.2, "volume": 100},
            index=times,
        )
        continuation.loc[times[-1], "low"] = 104.0
        one = pd.concat([one, continuation])
        result = backtest(one, five, hour, day, sessions=1)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["trades_detail"][0]["reason"], "stop")
        self.assertEqual(result["trades_detail"][0]["exit_time"], times[-1].isoformat())
        self.assertLess(result["trades_detail"][0]["entry_time"], "2026-11-11T11:00:00-05:00")

    def test_no_fill_at_or_after_11_am(self):
        one, five, hour, day, _ = bullish_case()
        one.index += pd.Timedelta(hours=1)
        five.index += pd.Timedelta(hours=1)
        next_time = one.index[-1] + pd.Timedelta(minutes=1)
        one.loc[next_time, ["open", "high", "low", "close", "volume"]] = [105.2, 105.3, 105.1, 105.2, 100]
        self.assertEqual(backtest(one, five, hour, day, sessions=1)["trades"], 0)

    def test_missing_next_minute_does_not_fill_late(self):
        one, five, hour, day, _ = bullish_case()
        late = one.index[-1] + pd.Timedelta(minutes=2)
        one.loc[late, ["open", "high", "low", "close", "volume"]] = [105.2, 105.3, 104, 104.7, 100]
        self.assertEqual(backtest(one, five, hour, day, sessions=1)["trades"], 0)

    def test_zone_must_exist_when_signal_is_observed(self):
        one, five, hour, day, _ = bullish_case()
        next_time = one.index[-1] + pd.Timedelta(minutes=1)
        one.loc[next_time, ["open", "high", "low", "close", "volume"]] = [105.21, 105.3, 104, 104.7, 100]
        signal_time = next_time
        zone = {"ticker": "SPY", "kind": "support", "low": 104, "high": 105,
                "available_at": (signal_time + pd.Timedelta(seconds=30)).isoformat()}
        result = backtest(one, five, hour, day, sessions=1,
                          zone_history=validate_zone_history([zone]))
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["trades_detail"][0]["marked_zones_at_signal"], [])
        zone["available_at"] = (signal_time - pd.Timedelta(seconds=30)).isoformat()
        result = backtest(one, five, hour, day, sessions=1,
                          zone_history=validate_zone_history([zone]))
        self.assertEqual(result["trades_detail"][0]["marked_zones_at_signal"][0]["kind"], "support")
        self.assertTrue(any(level["kind"] == "prior_day_high" for level in
                            result["trades_detail"][0]["mapped_zones_at_signal"]))


if __name__ == "__main__":
    unittest.main()
