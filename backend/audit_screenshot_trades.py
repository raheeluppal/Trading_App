"""Point-in-time audit of the dated screenshot setups.

This is an evaluation tool, not a strategy tuner: cases and directions are
declared before replay, and the normal evaluator is used without changing
thresholds per ticker or date.
"""

import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.timeframe import TimeFrame

from backtest_playbook import fetch
from playbook import entry_eligible, evaluate_playbook


ET = ZoneInfo("America/New_York")
CASES = [
    ("MSFT", "2026-06-03", "PUT"),
    ("TSLA", "2026-06-08", "BUY"),
    ("MU", "2026-06-09", "PUT"),
    ("MU", "2026-06-10", "PUT"),
    ("MU", "2026-06-12", "BUY"),
    ("MU", "2026-06-16", "PUT"),
    ("MU", "2026-07-16", "BUY"),
    ("SPY", "2026-06-24", "BUY"),
    ("SPY", "2026-07-24", "BUY"),
    ("QQQ", "2026-07-20", "PUT"),
    ("TSLA", "2026-07-20", "PUT"),
    ("TSLA", "2026-07-21", "BUY"),
    ("DDOG", "2026-11-17", "PUT"),
]


def audit_case(ticker, session_date, expected):
    day = datetime.strptime(session_date, "%Y-%m-%d").replace(tzinfo=ET)
    start = day - timedelta(days=8)
    end = day + timedelta(days=1, hours=1)
    minute = fetch(ticker, TimeFrame.Minute, start, end)
    five = fetch(ticker, TimeFrame(5, TimeFrame.Minute), start, end)
    hourly = fetch(ticker, TimeFrame.Hour, day - timedelta(days=45), end)
    daily = fetch(ticker, TimeFrame.Day, day - timedelta(days=150), end)
    benchmark = fetch("SPY" if ticker == "QQQ" else "QQQ", TimeFrame(5, TimeFrame.Minute), start, end)
    minute = minute.between_time("09:30", "15:59")
    signals = []
    used = set()
    target_date = day.date()
    for i, (bar_time, _) in enumerate(minute.iterrows()):
        if bar_time.date() != target_date or i + 1 >= len(minute):
            continue
        next_time = minute.index[i + 1]
        if next_time != bar_time + pd.Timedelta(minutes=1) or not entry_eligible(next_time.to_pydatetime()):
            continue
        now = bar_time.to_pydatetime() + timedelta(minutes=1)
        five_end = five.index.searchsorted(now)
        hour_end = hourly.index.searchsorted(now)
        daily_end = daily.index.searchsorted(now)
        market_end = benchmark.index.searchsorted(now)
        setup = evaluate_playbook(
            minute.iloc[: i + 1],
            five.iloc[max(0, five_end - 234) : five_end],
            hourly.iloc[max(0, hour_end - 60) : hour_end],
            daily.iloc[max(0, daily_end - 60) : daily_end],
            now=now,
            market_five_bars=benchmark.iloc[max(0, market_end - 234) : market_end],
        )
        if setup.signal not in {"BUY", "PUT"} or setup.signal_bar in used:
            continue
        used.add(setup.signal_bar)
        signals.append({
            "time": next_time.isoformat(),
            "signal": setup.signal,
            "setup": setup.name,
            "quality_score": setup.quality_score,
            "location": setup.location,
        })
    caught = any(signal["signal"] == expected for signal in signals)
    strict_caught = any(signal["signal"] == expected and signal["quality_score"] >= 5 for signal in signals)
    return {
        "ticker": ticker,
        "date": session_date,
        "expected": expected,
        "result": "CAUGHT" if caught else "MISSED",
        "strict_result": "CAUGHT" if strict_caught else "MISSED",
        "signals": signals,
    }


def main():
    results = [audit_case(*case) for case in CASES]
    print(json.dumps({
        "method": "fixed screenshot cases; unchanged evaluator; point-in-time replay; strict_result means quality>=5",
        "cases": results,
        "caught": sum(item["result"] == "CAUGHT" for item in results),
        "missed": sum(item["result"] == "MISSED" for item in results),
        "strict_caught": sum(item["strict_result"] == "CAUGHT" for item in results),
        "strict_missed": sum(item["strict_result"] == "MISSED" for item in results),
    }, indent=2))


if __name__ == "__main__":
    main()
