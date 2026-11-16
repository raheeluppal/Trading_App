"""Broker bar shaping and session VWAP agree with strategy inputs."""

import unittest

import pandas as pd

from features import get_chart_data


class ChartFeatureTests(unittest.TestCase):
    def test_multiindex_bars_and_daily_vwap_reset(self):
        stamps = pd.DatetimeIndex(["2026-11-10 14:30Z", "2026-11-10 14:31Z",
                                   "2026-11-11 14:30Z"])
        index = pd.MultiIndex.from_product([["SPY"], stamps], names=["symbol", "timestamp"])
        bars = pd.DataFrame({"open": [100, 102, 110], "high": [101, 103, 111],
                             "low": [99, 101, 109], "close": [100, 102, 110],
                             "volume": [10, 10, 10]}, index=index)
        result = get_chart_data(bars)
        self.assertEqual([bar["vwap"] for bar in result], [100, 101, 110])
        self.assertEqual(result[0]["timestamp"], stamps[0].isoformat())


if __name__ == "__main__":
    unittest.main()
