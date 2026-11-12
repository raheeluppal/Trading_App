"""Focused candle-level regressions for the visual playbook signal path."""

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from playbook import evaluate_playbook, in_entry_window


ET = ZoneInfo("America/New_York")


def bars(index, closes, volume=100, step=0.03):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": closes - step,
            "high": closes + step,
            "low": closes - step * 2,
            "close": closes,
            "volume": np.full(len(closes), volume),
        },
        index=index,
    )


def bullish_case():
    day_idx = pd.date_range("2026-09-01", periods=45, freq="B", tz="UTC")
    day = bars(day_idx, np.linspace(90, 110, len(day_idx)))
    hour_days = pd.date_range("2026-11-02", periods=8, freq="B", tz=ET)
    hour_idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in hour_days for h in range(10, 16)])
    hour = bars(hour_idx, np.linspace(99, 108, len(hour_idx)))
    five_idx = pd.date_range("2026-11-11 09:30", periods=25, freq="5min", tz=ET)
    five_idx = five_idx.append(pd.date_range("2026-11-10 14:00", periods=25, freq="5min", tz=ET)).sort_values()
    five = bars(five_idx, np.linspace(99, 105, len(five_idx)))
    five.loc[five_idx[-1], "open"] = 104.7
    five.loc[five_idx[-1], "close"] = 105.0
    ema9 = five["close"].ewm(span=9, adjust=False).mean().iloc[-1]
    minute_idx = pd.date_range("2026-11-11 09:30", periods=121, freq="min", tz=ET)
    one = bars(minute_idx, np.linspace(102, 104.8, len(minute_idx)))
    one.loc[minute_idx[-3], ["open", "low", "close", "high", "volume"]] = [ema9 + .01, ema9 - .03, ema9 + .08, ema9 + .12, 200]
    one.loc[minute_idx[-2], ["open", "low", "close", "high", "volume"]] = [ema9 + .08, ema9 + .04, ema9 + .14, ema9 + .17, 200]
    one.loc[minute_idx[-1], ["open", "low", "close", "high", "volume"]] = [ema9 + .2, ema9 + .19, 105.2, 105.24, 100]
    now = datetime(2026, 11, 11, 11, 32, tzinfo=ET)
    # Keep the same candle pattern, but place the trigger inside the morning window.
    one.index -= pd.Timedelta(hours=1)
    five.index -= pd.Timedelta(hours=1)
    now -= pd.Timedelta(hours=1)
    return one, five, hour, day, now


def mirror_prices(frame):
    bearish = frame.copy()
    bearish["open"] = 210 - frame["open"]
    bearish["high"] = 210 - frame["low"]
    bearish["low"] = 210 - frame["high"]
    bearish["close"] = 210 - frame["close"]
    return bearish


class PlaybookTests(unittest.TestCase):
    def test_entry_window_boundaries_are_eastern(self):
        self.assertFalse(in_entry_window(datetime(2026, 11, 11, 9, 29, tzinfo=ET)))
        self.assertTrue(in_entry_window(datetime(2026, 11, 11, 9, 30, tzinfo=ET)))
        self.assertTrue(in_entry_window(datetime(2026, 11, 11, 10, 59, tzinfo=ET)))
        self.assertFalse(in_entry_window(datetime(2026, 11, 11, 11, 0, tzinfo=ET)))

    def test_confirmed_pullback_requires_entry_break(self):
        one, five, hour, day, now = bullish_case()
        result = evaluate_playbook(one, five, hour, day, now)
        self.assertEqual(result.signal, "BUY")
        self.assertEqual(result.name, "EMA/VWAP pullback")
        self.assertLess(result.stop, result.entry)
        self.assertGreaterEqual(result.target - result.entry, 2 * (result.entry - result.stop))

        one.loc[one.index[-1], "close"] = one.loc[one.index[-2], "high"] - .01
        self.assertEqual(evaluate_playbook(one, five, hour, day, now).signal, "WAIT")

    def test_bearish_pullback_is_a_put_idea_not_a_stock_buy(self):
        one, five, hour, day, now = bullish_case()
        result = evaluate_playbook(*(mirror_prices(bars) for bars in (one, five, hour, day)), now)
        self.assertEqual(result.signal, "PUT")
        self.assertGreater(result.stop, result.entry)
        self.assertLess(result.target, result.entry)

    def test_weak_volume_and_stale_bars_do_not_signal(self):
        one, five, hour, day, now = bullish_case()
        one.loc[one.index[-3:-1], "volume"] = 10
        self.assertEqual(evaluate_playbook(one, five, hour, day, now).signal, "WAIT")
        one, five, hour, day, now = bullish_case()
        self.assertEqual(
            evaluate_playbook(one, five, hour, day, now.replace(hour=12)).signal,
            "WAIT",
        )

    def test_missing_context_and_premarket_do_not_signal(self):
        one, five, hour, day, now = bullish_case()
        self.assertEqual(evaluate_playbook(one, five, hour.iloc[:3], day, now).signal, "WAIT")
        self.assertEqual(
            evaluate_playbook(one, five, hour, day, now.replace(hour=9, minute=20)).signal,
            "WAIT",
        )

    def test_11_am_cutoff_rejects_a_confirmed_setup(self):
        one, five, hour, day, now = bullish_case()
        self.assertEqual(evaluate_playbook(one, five, hour, day, now).signal, "BUY")
        result = evaluate_playbook(one, five, hour, day, now.replace(hour=11, minute=0))
        self.assertEqual(result.signal, "WAIT")
        self.assertIn("09:30-11:00", result.reason)


if __name__ == "__main__":
    unittest.main()
