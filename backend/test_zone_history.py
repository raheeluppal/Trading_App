"""Historical zone activation and validation are point-in-time."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from playbook import ET
from zone_history import active_zones, load_zone_database, load_zone_file, validate_zone_history


class ZoneHistoryTests(unittest.TestCase):
    def test_activation_and_removal_boundaries(self):
        history = validate_zone_history([{
            "ticker": "SPY", "kind": "supply", "low": 100, "high": 101,
            "available_at": "2026-10-01T10:00:00-04:00",
            "removed_at": "2026-10-01T10:30:00-04:00",
        }])
        self.assertEqual(active_zones(history, "SPY", datetime(2026, 10, 1, 9, 59, tzinfo=ET)), [])
        self.assertEqual(len(active_zones(history, "SPY", datetime(2026, 10, 1, 10, tzinfo=ET))), 1)
        self.assertEqual(active_zones(history, "SPY", datetime(2026, 10, 1, 10, 30, tzinfo=ET)), [])
        self.assertEqual(active_zones(history, "QQQ", datetime(2026, 10, 1, 10, tzinfo=ET)), [])

    def test_unknown_or_naive_annotation_rejected(self):
        base = {"ticker": "SPY", "kind": "support", "low": 100, "high": 101}
        with self.assertRaisesRegex(ValueError, "timezone"):
            validate_zone_history([{**base, "available_at": "2026-10-01T10:00:00"}])
        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate_zone_history([{**base, "ticker": "AMZN", "available_at": "2026-10-01T10:00:00-04:00"}])

    def test_file_and_missing_db(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "zones.json"
            path.write_text(json.dumps([{
                "ticker": "SPY", "kind": "demand", "low": 100, "high": 101,
                "available_at": "2026-10-01T10:00:00-04:00",
            }]), encoding="utf-8")
            self.assertEqual(load_zone_file(path)[0]["kind"], "demand")
            self.assertEqual(load_zone_database(Path(folder) / "missing.db"), [])


if __name__ == "__main__":
    unittest.main()
