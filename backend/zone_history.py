"""Point-in-time marked zones shared by live signals and historical replay."""

import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from playbook import ET
from universe import TRADE_TICKERS


ZONE_KINDS = frozenset({
    "supply", "demand", "support", "resistance", "trendline_support",
    "trendline_resistance", "ath_retest", "daily_gap",
})


def _timestamp(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be an ISO-8601 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid {field}: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")
    return parsed.astimezone(ET)


def validate_zone_history(rows):
    """Normalize annotations; reject unknown or retrospective timestamps."""
    if not isinstance(rows, list):
        raise ValueError("Zone history must be a JSON array")
    normalized = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Zone #{index + 1} must be an object")
        ticker = row.get("ticker")
        kind = row.get("kind")
        if ticker not in TRADE_TICKERS or kind not in ZONE_KINDS:
            raise ValueError(f"Zone #{index + 1} has an unsupported ticker or kind")
        try:
            low, high = float(row["low"]), float(row["high"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Zone #{index + 1} needs numeric low/high") from error
        if not all(math.isfinite(value) for value in (low, high)) or low <= 0 or high < low:
            raise ValueError(f"Zone #{index + 1} has invalid price bounds")
        available = _timestamp(row.get("available_at", row.get("created_at")), "available_at")
        removed_raw = row.get("removed_at")
        removed = _timestamp(removed_raw, "removed_at") if removed_raw else None
        if removed is not None and removed < available:
            raise ValueError(f"Zone #{index + 1} was removed before it was available")
        normalized.append({
            "ticker": ticker, "kind": kind, "low": low, "high": high,
            "available_at": available, "removed_at": removed,
        })
    return sorted(normalized, key=lambda zone: zone["available_at"])


def active_zones(history, ticker, when):
    """Only levels already recorded and not yet removed at `when`."""
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError("Replay time must be timezone-aware")
    now = when.astimezone(ET)
    return [
        {"kind": zone["kind"], "low": zone["low"], "high": zone["high"]}
        for zone in history
        if zone["ticker"] == ticker and zone["available_at"] <= now
        and (zone["removed_at"] is None or now < zone["removed_at"])
    ]


def load_zone_file(path):
    with Path(path).open(encoding="utf-8") as handle:
        return validate_zone_history(json.load(handle))


def load_zone_database(path):
    db_path = Path(path)
    if not db_path.is_file():
        return []
    with closing(sqlite3.connect(db_path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(marked_zones)")}
        if not columns:
            return []
        if not {"ticker", "kind", "low", "high", "created_at"}.issubset(columns):
            raise ValueError("Marked-zone table has an unsupported schema")
        removed = "removed_at" if "removed_at" in columns else "NULL AS removed_at"
        rows = connection.execute(
            f"SELECT ticker, kind, low, high, created_at, {removed} FROM marked_zones"
        ).fetchall()
    return validate_zone_history([
        {"ticker": ticker, "kind": kind, "low": low, "high": high,
         "available_at": created, "removed_at": removed_at}
        for ticker, kind, low, high, created, removed_at in rows
    ])
