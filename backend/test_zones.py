"""Marked price zones persist and are restricted to the six-symbol universe."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import main
from zone_history import load_zone_database


class ZoneApiTests(unittest.TestCase):
    def test_zone_round_trip(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(main, "DB_PATH", Path(folder) / "signals.db"):
            main.init_history_db()
            created = main.add_marked_zone("SPY", main.MarkedZone(kind="demand", low=499, high=501))
            self.assertEqual(created["ticker"], "SPY")
            self.assertEqual(main.get_marked_zones("SPY")["zones"][0]["kind"], "demand")
            self.assertTrue(main.delete_marked_zone("SPY", created["id"])["deleted"])
            self.assertEqual(main.get_marked_zones("SPY")["zones"], [])
            history = load_zone_database(main.DB_PATH)
            self.assertEqual(len(history), 1)
            self.assertIsNotNone(history[0]["removed_at"])

    def test_invalid_zone_and_ticker(self):
        with self.assertRaises(HTTPException):
            main.add_marked_zone("AMZN", main.MarkedZone(kind="support", low=1, high=2))
        with self.assertRaises(HTTPException):
            main.add_marked_zone("SPY", main.MarkedZone(kind="momentum", low=1, high=2))
        with self.assertRaises(HTTPException):
            main.add_marked_zone("SPY", main.MarkedZone(kind="support", low=2, high=1))


if __name__ == "__main__":
    unittest.main()
