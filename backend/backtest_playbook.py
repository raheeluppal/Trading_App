"""Historical stock-share replay of the live visual-playbook signal rules.

BUY ideas buy stock; PUT ideas short stock. This does not model options,
borrow fees or availability, margin, partial exits, or real broker fills.
"""

import argparse
import json
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data_feed import STOCK_DATA_FEED, data_client
from playbook import evaluate_playbook, in_entry_window
from universe import TRADE_TICKERS


ET = ZoneInfo("America/New_York")
OHLCV = ["open", "high", "low", "close", "volume"]


def fetch(ticker, timeframe, start, end):
    if data_client is None:
        raise RuntimeError("Alpaca historical data client is unavailable")
    raw = data_client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=STOCK_DATA_FEED,
        )
    ).df
    if raw is None or raw.empty:
        raise RuntimeError(f"No historical {timeframe} bars returned for {ticker}")
    frame = raw.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame.index = frame.index.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(ET)
    return frame[OHLCV].astype(float).sort_index()


def _exit_price(side, bar, stop, target):
    """Conservative intrabar handling: the stop wins a both-hit candle."""
    if side == "BUY":
        if bar["low"] <= stop:
            return min(float(bar["open"]), stop), "stop"
        if bar["high"] >= target:
            return target, "target"
    else:
        if bar["high"] >= stop:
            return max(float(bar["open"]), stop), "stop"
        if bar["low"] <= target:
            return target, "target"
    return None, None


def _stock_result(side, entry, exit_price, cost_bps, shares):
    """Mark a long/short share trade with an assumed round-trip notional cost."""
    direction = 1 if side == "BUY" else -1
    gross_per_share = direction * (exit_price - entry)
    net_per_share = gross_per_share - entry * cost_bps / 10000
    return {
        "stock_side": "LONG" if direction == 1 else "SHORT",
        "shares": shares,
        "net_pnl_dollars": round(net_per_share * shares, 4),
        "net_return_pct": round(net_per_share / entry * 100, 4),
    }


def backtest(minute, five, hourly, daily, sessions=5, round_trip_cost_bps=8, shares=1):
    if shares < 1 or round_trip_cost_bps < 0:
        raise ValueError("shares must be positive and round-trip cost nonnegative")
    minute = minute.between_time("09:30", "15:59")
    five = five.between_time("09:30", "15:55")
    dates = sorted(set(minute.index.date))[-sessions:]
    if not dates:
        raise RuntimeError("No regular-session minute bars to test")
    first_test = dates[0]
    trades = []
    position = None
    pending = None
    used_signals = set()
    for i, (bar_time, bar) in enumerate(minute.iterrows()):
        if bar_time.date() < first_test:
            continue
        now = bar_time.to_pydatetime() + timedelta(minutes=1)
        if pending is not None:
            if bar_time.date() == pending["date"] and in_entry_window(bar_time.to_pydatetime()):
                fill = float(bar["open"])
                risk = fill - pending["stop"] if pending["side"] == "BUY" else pending["stop"] - fill
                reward = pending["target"] - fill if pending["side"] == "BUY" else fill - pending["target"]
                if risk > 0 and reward > 0:
                    position = dict(pending, entry=fill, entry_time=bar_time.isoformat())
            pending = None
        if position is not None:
            exit_price, reason = _exit_price(position["side"], bar, position["stop"], position["target"])
            if exit_price is None and position.get("structure_exit"):
                exit_price, reason = float(bar["open"]), "5m_structure"
            last_bar_of_day = i + 1 == len(minute) or minute.index[i + 1].date() != bar_time.date()
            if exit_price is None and last_bar_of_day:
                exit_price, reason = float(bar["close"]), "session_close"
            if exit_price is None:
                completed_five = five.index[five.index + pd.Timedelta(minutes=5) == bar_time + pd.Timedelta(minutes=1)]
                if len(completed_five):
                    j = five.index.get_loc(completed_five[-1])
                    if j > 0:
                        current, previous = five.iloc[j], five.iloc[j - 1]
                        position["structure_exit"] = (
                            current["low"] < previous["low"] if position["side"] == "BUY"
                            else current["high"] > previous["high"]
                        )
            if exit_price is not None:
                stock_result = _stock_result(position["side"], position["entry"], exit_price, round_trip_cost_bps, shares)
                risk = abs(position["entry"] - position["stop"])
                trades.append({
                    "side": position["side"], "setup": position["setup"],
                    "entry_time": position["entry_time"], "exit_time": bar_time.isoformat(),
                    "entry": round(position["entry"], 4), "exit": round(exit_price, 4),
                    "reason": reason, **stock_result,
                    "net_r": round(stock_result["net_pnl_dollars"] / shares / risk, 4) if risk else None,
                })
                position = None
        if position is not None or pending is not None or i + 1 >= len(minute):
            continue
        next_time = minute.index[i + 1]
        if next_time.date() != bar_time.date() or not in_entry_window(next_time.to_pydatetime()):
            continue
        # The live evaluator only needs recent intraday history; keep replay
        # work bounded without discarding the current session's VWAP history.
        setup = evaluate_playbook(minute.iloc[max(0, i - 2000) : i + 1], five, hourly, daily, now=now)
        key = (setup.signal, setup.signal_bar)
        if setup.signal not in {"BUY", "PUT"} or key in used_signals:
            continue
        used_signals.add(key)
        pending = {
            "side": setup.signal, "setup": setup.name, "stop": setup.stop,
            "target": setup.target, "date": next_time.date(),
        }

    # A data gap after the final minute should not manufacture a profitable exit.
    if position is not None:
        last_time = minute.index[-1]
        last_close = float(minute.iloc[-1]["close"])
        stock_result = _stock_result(position["side"], position["entry"], last_close, round_trip_cost_bps, shares)
        risk = abs(position["entry"] - position["stop"])
        trades.append({
            "side": position["side"], "setup": position["setup"],
            "entry_time": position["entry_time"], "exit_time": last_time.isoformat(),
            "entry": round(position["entry"], 4), "exit": round(last_close, 4),
            "reason": "data_end", **stock_result,
            "net_r": round(stock_result["net_pnl_dollars"] / shares / risk, 4) if risk else None,
        })

    returns = pd.Series([t["net_return_pct"] / 100 for t in trades], dtype=float)
    equity = (1 + returns).cumprod()
    equity_with_start = pd.concat([pd.Series([1.0]), equity], ignore_index=True)
    drawdown = equity_with_start / equity_with_start.cummax() - 1
    return {
        "sessions": len(dates), "first_session": str(dates[0]), "last_session": str(dates[-1]),
        "minute_bars": int(sum(minute.index.date >= first_test)),
        "trades": len(trades), "bullish": sum(t["side"] == "BUY" for t in trades),
        "bearish": sum(t["side"] == "PUT" for t in trades),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 2) if len(returns) else None,
        "mean_trade_return_pct": round(float(returns.mean() * 100), 4) if len(returns) else None,
        "compounded_return_pct": round(float((equity.iloc[-1] - 1) * 100), 4) if len(equity) else 0.0,
        "max_trade_equity_drawdown_pct": round(float(drawdown.min() * 100), 4) if len(drawdown) else 0.0,
        "round_trip_cost_bps": round_trip_cost_bps,
        "shares_per_trade": shares,
        "total_net_pnl_dollars": round(sum(t["net_pnl_dollars"] for t in trades), 4),
        "trades_detail": trades,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=8.0)
    parser.add_argument("--shares", type=int, default=1)
    args = parser.parse_args()
    if args.sessions < 1 or args.cost_bps < 0 or args.shares < 1:
        parser.error("sessions and shares must be positive and cost-bps nonnegative")
    end = datetime.now(ET)
    if end.time() < time(16, 0):
        end = datetime.combine(end.date(), time(0), ET)
    start_intraday = end - timedelta(days=max(14, args.sessions * 3))
    ticker = args.ticker.upper()
    if ticker not in TRADE_TICKERS:
        parser.error(f"ticker must be one of: {', '.join(TRADE_TICKERS)}")
    minute = fetch(ticker, TimeFrame.Minute, start_intraday, end)
    five = fetch(ticker, TimeFrame(5, TimeFrame.Minute), start_intraday, end)
    hourly = fetch(ticker, TimeFrame.Hour, end - timedelta(days=60), end)
    daily = fetch(ticker, TimeFrame.Day, end - timedelta(days=120), end)
    result = backtest(minute, five, hourly, daily, args.sessions, args.cost_bps, args.shares)
    result.update({
        "ticker": ticker,
        "engine": "visual_playbook_rules",
        "feed": str(STOCK_DATA_FEED.value),
        "instrument": "stock_shares",
        "short_borrow_costs_included": False,
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
