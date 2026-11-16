# Quick start

Use Python 3.12 and install the packages in [requirements.txt](./requirements.txt). Set `ALPACA_API_KEY` and `ALPACA_API_SECRET` in the environment or a local `.env` file; never paste keys into source files. Alpaca paper trading is recommended while validating this strategy.

From `backend/`, start the API with `uvicorn main:app --port 8000`. From `frontend/`, run `npm install` and `npm start`. The dashboard is at `http://localhost:3000`.

Only QQQ, SPY, TSLA, AAPL, NVDA, and MSFT are supported. New share positions require a confirmed playbook setup from 9:30 to before 11:00 a.m. Eastern; orders that reduce a position remain allowed later. The app does not automatically execute or manage exits. The chart lets you enter manually identified supply/demand, trendline, ATH-retest, and other price zones; those affect current signals.

To replay all six symbols over roughly two months, run `python backtest.py` from `backend/`. Timestamped marked zones already saved in the local database are replayed only after they were recorded. Use `--zones-file zones.json` for older historical zone annotations; current marks cannot be applied to periods before they existed. See [README.md](./README.md) for the file format and execution assumptions.
