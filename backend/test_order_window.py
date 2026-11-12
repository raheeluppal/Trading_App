"""Manual order boundaries: outside the morning window, only risk reduction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from main import place_order_endpoint


class ManualOrderWindowTests(unittest.TestCase):
    def test_outside_window_rejects_new_positions(self):
        with patch("main.in_entry_window", return_value=False), patch("main.get_position", return_value=None), \
             patch("main.place_buy_order") as buy, patch("main.place_sell_order") as sell:
            self.assertFalse(place_order_endpoint("SPY", 1, "BUY")["success"])
            self.assertFalse(place_order_endpoint("SPY", 1, "SELL")["success"])
            buy.assert_not_called()
            sell.assert_not_called()

    def test_outside_window_allows_only_reducing_existing_position(self):
        order = SimpleNamespace(id="paper-order", status="accepted")
        long_position = SimpleNamespace(side=SimpleNamespace(value="long"), qty="10")
        short_position = SimpleNamespace(side=SimpleNamespace(value="short"), qty="10")
        with patch("main.in_entry_window", return_value=False), patch("main.get_position") as position, \
             patch("main.place_buy_order", return_value=order) as buy, \
             patch("main.place_sell_order", return_value=order) as sell:
            position.return_value = long_position
            self.assertTrue(place_order_endpoint("SPY", 5, "SELL")["success"])
            self.assertFalse(place_order_endpoint("SPY", 11, "SELL")["success"])
            self.assertFalse(place_order_endpoint("SPY", 1, "BUY")["success"])
            position.return_value = short_position
            self.assertTrue(place_order_endpoint("SPY", 5, "BUY")["success"])
            self.assertFalse(place_order_endpoint("SPY", 11, "BUY")["success"])
            self.assertFalse(place_order_endpoint("SPY", 1, "SELL")["success"])
            buy.assert_called_once_with("SPY", 5)
            sell.assert_called_once_with("SPY", 5)


if __name__ == "__main__":
    unittest.main()
