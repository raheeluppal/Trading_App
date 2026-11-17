# Intraday stock playbook

This app monitors the supplied *Intraday Trading Visual Field Guide* as explicit candle rules, not a trained prediction model. It supports only QQQ, SPY, TSLA, AAPL, NVDA, and MSFT. Signals are candidate stock trades: `BUY` is long shares, `PUT` is short shares (the older signal label is retained for API compatibility). The guide itself discusses options, so this is a stock-share interpretation, not an options implementation.

The existing API/database field named `probability` is binary (`1` = setup confirmed, `0` = not confirmed) for compatibility; it is not a success probability or confidence score.

New entries require a current matching setup and normally occur from 9:30 a.m. up to, but not including, 11:00 a.m. Eastern. A separate gap-and-go continuation setup may enter from 11:00 a.m. up to, but not including, 2:30 p.m. Eastern when the 5-minute trend, VWAP/EMA structure, breakdown, volume, and mapped target all align. Position-reducing orders can be submitted later; there is no 11 a.m. forced exit. The app never submits automatic orders or exits. Stops, targets, and 5-minute structure exits are modeled by the backtest, but the live broker position page does not pretend to know or manage a stop/target for an existing position. The manual order endpoint submits market orders and does not guarantee a signal price or stop fill.

The evaluator uses completed daily and 1-hour trend context, 5-minute confirmation, a completed 2-minute signal candle with stronger volume, a subsequent 1-minute entry break, logical stops, and a mapped target. It checks EMA/VWAP pullbacks, level rejections, range bounces, break/retests, and major breakdowns. Automatically mapped levels include prior-day high/low, 1-hour extremes, recent 5-minute range edges, EMA 9/21, session VWAP, and the daily opening gap. Hand-drawn supply/demand, trendline, ATH-retest, and other price zones cannot be verified from candles alone: enter their low/high bounds on the chart or via `POST /zones/{ticker}`. Marked zones affect live signals immediately. Backtests use a marked zone only after the timestamp when it was actually recorded, and stop using it when removed. A marked ATH area is user-supplied and is not independently verified as an all-time high.

The guide does not specify numeric daily-loss, trades-per-day, position-size, or fixed risk/reward limits, so none are presented as strategy rules. The order endpoint requires a positive whole-share quantity. Broker margin, short-borrow availability, order fills, and risk management remain the operator’s responsibility.

## Run

Install `requirements.txt`, set `ALPACA_API_KEY` and `ALPACA_API_SECRET`, then run `uvicorn main:app --port 8000` from `backend/`. Run `npm install` and `npm start` from `frontend/`. If Alpaca data is unavailable, signals show `NO_DATA` and charts are empty; synthetic prices are never substituted.

## Backtest

From `backend/`, run `python backtest.py` for all six symbols over the latest 43 completed sessions (about two months), or `python backtest.py --ticker SPY --sessions 5` for a focused replay. Add `--summary-only` to omit individual trades. The default feed is Alpaca IEX; `ALPACA_DATA_FEED=sip` requires a SIP subscription. The replay calls the same playbook evaluator as the live monitor after each completed minute. An eligible signal is filled at the *immediately following* 1-minute open, only before 11 a.m.; a missing next minute cancels it. Stop-first handling applies when stop and target are both touched in one candle. Positions may exit later on stop, target, 5-minute structure, or session close. Each trade reports the stop, target, mapped zones, and marked zones visible when its signal appeared.

By default, historical marked zones come from the local signal database. To supply **historical** manually marked zones, run `python backtest.py --zones-file zones.json`. Each JSON entry needs the ticker, kind, price bounds, and the timezone-aware instant when that level was actually known. `removed_at` is optional. For example:

```json
[
  {"ticker":"SPY","kind":"supply","low":504.50,"high":505.25,"available_at":"2026-10-01T09:20:00-04:00","removed_at":"2026-10-02T16:00:00-04:00"}
]
```

The file replaces database zone history for that run. Current marks are **not** projected backward into earlier dates. If no historical marks exist, the replay still includes all automatically derived levels, but cannot claim to reproduce hand-drawn zones that were never recorded.

Results are simulated stock-share P&L with configurable assumed round-trip costs. A `PUT` signal is modeled as a short stock position. Borrow availability, borrow fees, margin, spread, slippage, latency, partial fills, and shared account buying-power constraints are not modeled. The app does not actually place protective exits: the replay's stop/target/5-minute/session exits represent a disciplined execution assumption, not a verified recording of live manual orders. In `ALL` mode, net dollars are summed across tickers; no portfolio percentage return is claimed.

The replay defaults to a minimum setup quality score of 5/6. The score rewards mapped support/resistance location, directional context, confirmed trigger, relative volume, at least 1.5R to the mapped target, and benchmark-index alignment when SPY/QQQ data is available. Use `--min-quality-score 0` through `6` to inspect broader or more selective candidate sets. A 6/6 run is intentionally ultra-selective and should not be treated as statistically reliable when it produces only a handful of trades.

Run the suite with `python -m unittest discover -p 'test_*.py'` from `backend/`.
