# Visual Day-Trading Playbook Monitor

This app monitors Alpaca stock candles and reports candidate setups from the supplied *Intraday Trading Visual Field Guide*. The live signal path uses explicit playbook rules, not the legacy XGBoost probability model. Signals are decision aids, not orders.

The monitored and manually tradable universe is limited to **QQQ, SPY, TSLA, AAPL, NVDA, and MSFT**. The chart shortcuts and backtest CLI use the same restriction. Legacy research artifacts may contain other symbols but do not drive live signals.

New playbook signals and manual orders that open or enlarge a stock position are allowed only from **9:30 a.m. to 11:00 a.m. Eastern** (11:00 excluded). Outside that window, manual orders may only reduce an existing position. The historical simulator closes any open position at the 10:59 one-minute bar's close; the live app does not automatically place or close playbook trades.

## Run

Install `requirements.txt`, set `ALPACA_API_KEY` and `ALPACA_API_SECRET` in your environment, then run `uvicorn main:app --reload --port 8000` from `backend/`. Run `npm install` and `npm start` from `frontend/` for the dashboard.

Without broker data, charts may show simulated candles for development, but `/signals` will return `NO_DATA`; simulated candles never produce playbook signals. The monitor checks once per minute. The dashboard polls more frequently but cannot create a new signal between monitor cycles.

## What the signal rules check

- Daily and 1-hour context use the 9/21 EMA relationship on completed bars. The 5-minute candle must agree with the entry direction.
- A 2-minute signal candle is assembled from two complete 1-minute candles during regular trading hours. Its volume must be at least 1.2 times the preceding 20 two-minute candles.
- The subsequent completed 1-minute price must cross the signal candle high for a bullish idea or low for a bearish idea. A stale price cannot trigger.
- The engine checks EMA/VWAP pullbacks, major-level rejection, range bounces, break/retests, and repeated-test breakdowns. Programmatic prior-candle support/resistance is used where the guide asks a human to mark a level; it cannot reproduce hand-drawn trendlines, supply/demand zones, gaps, ATHs, or option-chain selection.
- Each confirmed idea includes an observed entry, logical stop and the nearest mapped target in its direction. The mapped target must provide at least 2R under the current mechanical policy. `BUY` means a bullish/call idea and `PUT` means a bearish/put idea; neither is an order.

The numeric `probability` field is retained for older dashboard/API clients but is **binary** (`1` = rule-confirmed, `0` = not confirmed), not a calibrated probability. The `setup` field explains the result.

## Execution boundary

Automated entries are disabled. The old execution path placed stock market buys and ATR-based stops, which cannot implement the guide's call/put entries, option contract selection, 2-minute trigger prices, or 5-minute structure exits. Setting `LIVE_TRADING_ENABLED=1` does not enable playbook entries. Manual order endpoints remain separate from the strategy and should not be interpreted as its execution engine. Existing tracked positions still use their legacy position-management logic.

Before any automated trading is enabled, implement and verify option contract selection, broker order-state reconciliation, stop/target order handling, and 5-minute structure exits against paper data. Historical XGBoost training/retraining files remain for legacy research and are not used to issue live playbook signals.

Run the focused tests with `python -m unittest test_playbook -v` from `backend/`.

## Historical backtest

Run `python backtest.py --ticker SPY --sessions 5` from `backend/`. It fetches real Alpaca 1-minute, 5-minute, hourly and daily stock bars, then replays the same playbook evaluator at each completed minute. The default feed is IEX (`ALPACA_DATA_FEED=sip` requires a SIP subscription). Entries fill at the next 1-minute open only before 11:00 ET; stops and targets use intrabar highs/lows, with the stop assumed first if both are touched. A completed 5-minute structure break or the 10:59 window close also exits. The default round-trip cost is 8 basis points and can be set with `--cost-bps`.

Results are a **long/short underlying-share proxy**, not option P&L. They do not include option premiums, spreads, Greeks, partial exits, borrow constraints or realistic broker execution. A small number of signals is not evidence of a profitable strategy.
