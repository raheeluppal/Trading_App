"""Export the six-symbol visual-playbook stock replay as a charted PDF."""

import argparse
from datetime import datetime, time, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import pandas as pd
from alpaca.data.timeframe import TimeFrame

from backtest_playbook import ET, backtest, fetch
from data_feed import STOCK_DATA_FEED
from universe import TRADE_TICKERS
from zone_history import load_zone_database, load_zone_file


COLORS = {
    "prior_day_low": "#7952b3", "prior_day_high": "#7952b3",
    "hourly_support": "#2387b8", "hourly_resistance": "#2387b8",
    "range_support": "#777777", "range_resistance": "#777777",
    "vwap": "#16a6a6", "ema9": "#e3a11b", "ema21": "#c97524",
}


def _stamp(value):
    return pd.Timestamp(value).tz_convert(ET).strftime("%Y-%m-%d %H:%M")


def _summary_page(pdf, trades, results, first, last, cost_bps, shares, zone_source):
    fig = plt.figure(figsize=(14, 8.5), facecolor="white")
    fig.text(.06, .92, "VISUAL PLAYBOOK | STOCK-SHARE BACKTEST", fontsize=22, weight="bold")
    fig.text(.06, .865, f"{first} through {last}  |  Eastern Time  |  IEX/SIP feed: {STOCK_DATA_FEED.value}", fontsize=12)
    wins = sum(t["net_pnl_dollars"] > 0 for t in trades)
    pnl = sum(t["net_pnl_dollars"] for t in trades)
    fig.text(.06, .77, f"{len(trades)} simulated trades     {wins} wins / {len(trades)-wins} non-wins     Net P&L ${pnl:+.4f}", fontsize=17, weight="bold")
    fig.text(.06, .715, f"{shares} share(s) per trade  |  {cost_bps:g} bps assumed round-trip cost  |  Entries 09:30-10:59 ET; exits can occur later", fontsize=11)
    rows = [[ticker, str(result["trades"]), str(sum(t["net_pnl_dollars"] > 0 for t in result["trades_detail"])),
             f'${result["total_net_pnl_dollars"]:+.4f}', str(result["zone_annotations_available"])]
            for ticker, result in results.items()]
    ax = fig.add_axes([.06, .36, .88, .30]); ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Ticker", "Trades", "Winners", "Net P&L", "Historical zone annotations"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.7)
    for (r, _), cell in table.get_celld().items():
        if r == 0: cell.set_facecolor("#e7eef7")
    fig.text(.06, .27, "Important limitations", fontsize=13, weight="bold")
    note = ("These are replayed stock trades, not orders actually placed or broker-confirmed fills. "
            "Short trades assume stock borrow is available; borrow fees, slippage beyond the stated cost, "
            "partial fills, margin, and market impact are not modeled. Stops/targets and structure exits "
            "are simulated; the live app does not automatically execute or manage them. "
            "When both stop and target are touched in one bar, the stop wins. "
            f"Historical hand-marked zone source: {zone_source}. "
            "No manually drawn zones are invented; each chart lists the automatic zones mapped at signal time.")
    import textwrap
    fig.text(.06, .23, textwrap.fill(note, 145), fontsize=10, va="top", linespacing=1.7)
    pdf.savefig(fig); plt.close(fig)


def _ledger_pages(pdf, trades):
    for start in range(0, len(trades), 13):
        fig = plt.figure(figsize=(14, 8.5), facecolor="white")
        fig.text(.05, .92, "TRADE LEDGER | all timestamps ET", fontsize=19, weight="bold")
        rows = []
        for number, t in enumerate(trades[start:start+13], start+1):
            rows.append([str(number), t["ticker"], t["stock_side"], _stamp(t["entry_time"]), _stamp(t["exit_time"]),
                         f'${t["entry"]:.2f}', f'${t["exit"]:.2f}', t["reason"],
                         f'${t["net_pnl_dollars"]:+.4f}'])
        ax = fig.add_axes([.04, .12, .92, .75]); ax.axis("off")
        tab = ax.table(cellText=rows, colLabels=["#", "Ticker", "Side", "Entry", "Exit", "Entry $", "Exit $", "Exit reason", "Net P&L"],
                       loc="upper center", cellLoc="center", colWidths=[.035,.06,.06,.16,.16,.075,.075,.13,.10])
        tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 2.1)
        for (r, c), cell in tab.get_celld().items():
            if r == 0: cell.set_facecolor("#e7eef7")
            elif c == 8: cell.set_text_props(color="#16834a" if trades[start+r-1]["net_pnl_dollars"] > 0 else "#c13a3a")
        pdf.savefig(fig); plt.close(fig)


def _trade_page(pdf, number, trade, minute):
    entry = pd.Timestamp(trade["entry_time"]).tz_convert(ET)
    exit_at = pd.Timestamp(trade["exit_time"]).tz_convert(ET)
    day = minute.loc[minute.index.date == entry.date()].between_time("09:30", "15:59")
    chart = day.loc[max(day.index[0], entry - pd.Timedelta(minutes=30)):min(day.index[-1], exit_at + pd.Timedelta(minutes=10))]
    if chart.empty:
        raise RuntimeError(f'No minute bars for trade {number}: {trade["ticker"]} {entry}')
    fig = plt.figure(figsize=(14, 8.5), facecolor="white")
    ax = fig.add_axes([.07, .27, .66, .60])
    volume_ax = fig.add_axes([.07, .13, .66, .11], sharex=ax)
    fig.text(.07, .94, f'{number:02d}  {trade["ticker"]}  {trade["stock_side"]}  |  {trade["setup"]}', fontsize=17, weight="bold")
    fig.text(.07, .905, f'Entry {_stamp(entry)} ET @ ${trade["entry"]:.4f}   /   Exit {_stamp(exit_at)} ET @ ${trade["exit"]:.4f}   /   {trade["reason"]}', fontsize=10)
    fig.text(.07, .875, f'Net ${trade["net_pnl_dollars"]:+.4f} ({trade["net_return_pct"]:+.4f}%)   |   R {trade["net_r"]:+.3f}   |   Stop ${trade["stop"]:.4f}   Target ${trade["target"]:.4f}', fontsize=10)
    x = list(range(len(chart)))
    candle_colors = ["#26a269" if b["close"] >= b["open"] else "#c64646" for _, b in chart.iterrows()]
    for i, (_, b) in enumerate(chart.iterrows()):
        ax.vlines(i, b["low"], b["high"], color=candle_colors[i], linewidth=.7, zorder=2)
        ax.add_patch(Rectangle((i-.32, min(b["open"], b["close"])), .64,
                               max(abs(b["close"]-b["open"]), .0001), color=candle_colors[i], zorder=3))
    volume_ax.bar(x, chart["volume"], color=candle_colors, width=.7, alpha=.6)
    core = [chart["low"].min(), chart["high"].max(), trade["entry"], trade["exit"], trade["stop"], trade["target"]]
    lo, hi = min(core), max(core)
    pad = max((hi-lo)*.08, trade["entry"]*.0005)
    ax.set_ylim(lo-pad, hi+pad)
    visible = []
    zones = trade["mapped_zones_at_signal"] + trade["marked_zones_at_signal"]
    for zone in zones:
        low, high = float(zone["low"]), float(zone["high"])
        kind = zone["kind"]
        if high < lo-pad or low > hi+pad:
            continue
        if high > low:
            ax.axhspan(low, high, color="#a771d0" if zone in trade["marked_zones_at_signal"] else "#c8a9df", alpha=.13, zorder=0)
        else:
            ax.axhline(low, color=COLORS.get(kind, "#875aa8"), lw=.95, alpha=.65, zorder=1)
        visible.append(kind)
    for label, price, color, style in [("ENTRY", trade["entry"], "#173f9c", "--"),
                                        ("EXIT", trade["exit"], "#db681f", "--"),
                                        ("STOP", trade["stop"], "#bc1c38", ":"),
                                        ("TARGET", trade["target"], "#118454", ":")]:
        ax.axhline(price, color=color, linestyle=style, lw=1.5, label=f"{label} ${price:.4f}", zorder=4)
    entry_x = chart.index.get_indexer([entry], method="nearest")[0]
    exit_x = chart.index.get_indexer([exit_at], method="nearest")[0]
    ax.scatter([entry_x], [trade["entry"]], s=125, marker="^" if trade["stock_side"] == "LONG" else "v",
               color="#173f9c", edgecolor="white", zorder=6)
    ax.scatter([exit_x], [trade["exit"]], s=120, marker="X", color="#db681f", edgecolor="white", zorder=6)
    ax.annotate("ENTRY", (entry_x, trade["entry"]), xytext=(4, 10), textcoords="offset points", fontsize=8, color="#173f9c", weight="bold")
    ax.annotate("EXIT", (exit_x, trade["exit"]), xytext=(4, -15), textcoords="offset points", fontsize=8, color="#db681f", weight="bold")
    ax.set_xlim(-1, len(chart)); ax.grid(alpha=.18); ax.set_ylabel("Stock price ($)")
    ax.legend(loc="upper left", fontsize=7, ncol=2, framealpha=.85)
    volume_ax.set_ylabel("Volume", fontsize=8); volume_ax.tick_params(labelsize=7)
    ticks = sorted(set([0, *range(0, len(chart), max(1, len(chart)//7)), len(chart)-1]))
    volume_ax.set_xticks(ticks, [chart.index[i].strftime("%H:%M") for i in ticks])
    ax.tick_params(labelbottom=False)
    side = fig.add_axes([.76, .12, .22, .75]); side.axis("off")
    side.text(0, 1, "ZONES AT SIGNAL", fontsize=11, weight="bold", va="top")
    y = .95
    for zone in zones:
        low, high = float(zone["low"]), float(zone["high"])
        val = f"${low:.2f}" if low == high else f"${low:.2f}-${high:.2f}"
        label = zone["kind"].replace("_", " ")
        marker = "" if zone["kind"] in visible else " (off chart)"
        side.text(0, y, f"{label}\n{val}{marker}", fontsize=8.2, va="top", color=COLORS.get(zone["kind"], "#70409a"))
        y -= .061 if len(zones) > 10 else .074
    if not trade["marked_zones_at_signal"]:
        side.text(0, max(.02, y-.01), "No historical hand-marked\nzone at signal time.", fontsize=8, color="#9a4d29", va="top")
    fig.text(.07, .045, f'Location: {trade["location"]}  |  Automatic levels are signal-time snapshots; shaded bands are zones. Off-chart levels are listed at right.', fontsize=8)
    pdf.savefig(fig); plt.close(fig)


def export_report(path, sessions=43, cost_bps=8, shares=1, zones_file=None):
    end = datetime.now(ET)
    if end.time() < time(16, 0):
        end = datetime.combine(end.date(), time(0), ET)
    start = end - timedelta(days=max(14, (sessions * 7 + 4) // 5 + 20))
    source = zones_file or "local marked-zones database"
    history = load_zone_file(zones_file) if zones_file else load_zone_database(Path(__file__).resolve().parent / "signal_history.db")
    results, minutes = {}, {}
    for ticker in TRADE_TICKERS:
        print(f"Fetching and replaying {ticker}...", flush=True)
        minute = fetch(ticker, TimeFrame.Minute, start, end)
        five = fetch(ticker, TimeFrame(5, TimeFrame.Minute), start, end)
        hourly = fetch(ticker, TimeFrame.Hour, start - timedelta(days=35), end)
        daily = fetch(ticker, TimeFrame.Day, start - timedelta(days=120), end)
        results[ticker] = backtest(minute, five, hourly, daily, sessions, cost_bps, shares, ticker=ticker, zone_history=history)
        minutes[ticker] = minute
    trades = sorted([dict(ticker=ticker, **trade) for ticker, result in results.items()
                     for trade in result["trades_detail"]], key=lambda t: t["entry_time"])
    first = min(r["first_session"] for r in results.values())
    last = max(r["last_session"] for r in results.values())
    with PdfPages(path, metadata={"Title": "Visual Playbook Stock-Share Backtest", "Author": "Trading App"}) as pdf:
        _summary_page(pdf, trades, results, first, last, cost_bps, shares, source)
        _ledger_pages(pdf, trades)
        for n, trade in enumerate(trades, 1):
            _trade_page(pdf, n, trade, minutes[trade["ticker"]])
    return {"path": str(path), "trades": len(trades), "wins": sum(t["net_pnl_dollars"] > 0 for t in trades),
            "pnl": round(sum(t["net_pnl_dollars"] for t in trades), 4), "first": first, "last": last,
            "pages": 1 + (len(trades)+12)//13 + len(trades)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Trading_Playbook_Backtest_Report.pdf")
    parser.add_argument("--sessions", type=int, default=43)
    parser.add_argument("--cost-bps", type=float, default=8.0)
    parser.add_argument("--shares", type=int, default=1)
    parser.add_argument("--zones-file")
    args = parser.parse_args()
    if args.sessions < 1 or args.shares < 1 or args.cost_bps < 0:
        parser.error("sessions and shares must be positive; cost-bps must be nonnegative")
    print(export_report(args.output, args.sessions, args.cost_bps, args.shares, args.zones_file))
