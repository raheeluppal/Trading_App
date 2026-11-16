"""Marked price zones persist and are restricted to the six-symbol universe."""

import tempfile
import unittest
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import main
from zone_history import load_zone_database


class ZoneApiTests(unittest.TestCase):
    def test_existing_zone_table_is_migrated_without_losing_marks(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(main, "DB_PATH", Path(folder) / "signals.db"):
            with closing(sqlite3.connect(main.DB_PATH)) as connection, connection:
                connection.execute("CREATE TABLE marked_zones (id INTEGER PRIMARY KEY, ticker TEXT, kind TEXT, low REAL, high REAL, created_at TEXT)")
                connection.execute("INSERT INTO marked_zones VALUES (1, 'SPY', 'support', 100, 101, '2026-10-01T10:00:00-04:00')")
            main.init_history_db()
            self.assertEqual(main.get_marked_zones("SPY")["zones"][0]["id"], 1)
            self.assertEqual(len(load_zone_database(main.DB_PATH)), 1)

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
