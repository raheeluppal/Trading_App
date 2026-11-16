from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import time
import threading
import sqlite3
from contextlib import closing
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

from data_feed import (
    get_latest_bars,
    get_account_positions,
    place_buy_order, 
    place_sell_order, 
    get_account_info,
    get_position,
    get_orders,
)
from features import get_chart_data
from playbook import ET, evaluate_playbook, in_entry_window
from universe import TRADE_TICKERS
from zone_history import ZONE_KINDS

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


latest_signals = {}  # Current top-volume ticker signals only
latest_signal_universe = {}  # Signals for all tracked universe tickers
latest_volumes = {}  # Last observed per-ticker volume
top_volume_tickers = []  # Computed top N by latest volume
signal_history = {}  # Track signal history

TICKERS = TRADE_TICKERS
TOP_SIGNAL_COUNT = len(TICKERS)
DB_PATH = Path(__file__).resolve().parent / "signal_history.db"


class MarkedZone(BaseModel):
    kind: str
    low: float = Field(gt=0)
    high: float = Field(gt=0)


def load_marked_zones(ticker):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id, kind, low, high, created_at FROM marked_zones WHERE ticker = ? AND removed_at IS NULL ORDER BY id", (ticker,)).fetchall()
    return [dict(row) for row in rows]

for _ticker in TICKERS:
    signal_history[_ticker] = []

def init_history_db():
    """Initialize local SQLite database for signal history."""
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                probability REAL NOT NULL,
                signal TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_history_ticker_time
            ON signal_history (ticker, timestamp DESC)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS marked_zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                kind TEXT NOT NULL,
                low REAL NOT NULL,
                high REAL NOT NULL,
                created_at TEXT NOT NULL,
                removed_at TEXT
            )
        """)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(marked_zones)")}
        if "removed_at" not in columns:
            conn.execute("ALTER TABLE marked_zones ADD COLUMN removed_at TEXT")
        conn.commit()

def save_signal_record(ticker: str, timestamp: str, probability: float, signal: str):
    """Persist one signal record to SQLite."""
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute(
            "INSERT INTO signal_history (ticker, timestamp, probability, signal) VALUES (?, ?, ?, ?)",
            (ticker, timestamp, probability, signal)
        )
        conn.commit()


def init_trade_log_db():
    """Persistent completed-trade log (entry and exit narratives)."""
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                qty_shares INTEGER,
                pnl_dollars REAL,
                pnl_percent REAL,
                entry_source TEXT,
                entry_probability REAL,
                entry_threshold REAL,
                entry_reason_detail TEXT,
                exit_reason_code TEXT,
                exit_reason_detail TEXT,
                is_partial INTEGER NOT NULL DEFAULT 0,
                entry_order_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_trade_log_exit_time
            ON trade_log (exit_time DESC)
            """
        )
        conn.commit()


def load_history_from_db():
    """Load all historical signal records from SQLite into memory cache."""
    for ticker in TICKERS:
        signal_history[ticker] = []

    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT ticker, timestamp, probability, signal FROM signal_history ORDER BY timestamp ASC"
        ).fetchall()

    for ticker, timestamp, probability, signal in rows:
        if ticker not in signal_history:
            signal_history[ticker] = []
        signal_history[ticker].append({
            "timestamp": timestamp,
            "probability": float(probability),
            "signal": signal
        })

# -------------------------
# SIGNAL LOOP (every 60 sec)
# -------------------------
def evaluate_ticker(ticker):
    """Use only broker bars for playbook decisions; never synthetic fallback data."""
    one = get_latest_bars(ticker, minutes=3 * 390, interval="1m")
    five = get_latest_bars(ticker, minutes=3 * 390, interval="5m")
    hourly = get_latest_bars(ticker, minutes=60 * 60, interval="1h")
    daily = get_latest_bars(ticker, minutes=60 * 1440, interval="1d")
    if any(bars is None or bars.empty for bars in (one, five, hourly, daily)):
        return None, None
    return evaluate_playbook(one, five, hourly, daily, marked_zones=load_marked_zones(ticker)), one

def run_loop():
    global latest_signals, latest_signal_universe, latest_volumes, top_volume_tickers

    print("\nStarting playbook signal monitor (automatic entries disabled)...\n")
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking signals and managing positions...\n")
        
        for ticker in TICKERS:
            try:
                setup, data = evaluate_ticker(ticker)
                if setup is None:
                    latest_signal_universe[ticker] = {"probability": 0.0, "signal": "NO_DATA"}
                    continue

                # Latest completed stock price for display only.
                close_price = float(data.iloc[-1]['close'])
                
                latest_volumes[ticker] = int(data.iloc[-1]["volume"]) if "volume" in data.columns else 0
                
                prob = 1.0 if setup.signal in {"BUY", "PUT"} else 0.0

                # Store signal
                latest_signal_universe[ticker] = {
                    "probability": float(prob),
                    "signal": setup.signal,
                    "setup": setup.as_dict(),
                }

                # Signal-only: do not auto-submit orders.
                    
                # Track signal history for dashboard records
                record_timestamp = datetime.now().isoformat()
                record_signal = setup.signal
                record_probability = float(prob)

                signal_history[ticker].append({
                    "timestamp": record_timestamp,
                    "probability": record_probability,
                    "signal": record_signal
                })
                save_signal_record(
                    ticker=ticker,
                    timestamp=record_timestamp,
                    probability=record_probability,
                    signal=record_signal
                )
                
                print(f"  {ticker}: {setup.signal} ({setup.reason}) | Price: ${close_price:.2f}")
                    
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                latest_signal_universe[ticker] = {
                    "probability": 0.0,
                    "signal": "ERROR"
                }

        sorted_by_volume = sorted(
            latest_volumes.items(),
            key=lambda item: item[1],
            reverse=True
        )
        top_volume_tickers = [ticker for ticker, _ in sorted_by_volume[:TOP_SIGNAL_COUNT]]
        latest_signals = {
            ticker: latest_signal_universe[ticker]
            for ticker in top_volume_tickers
            if ticker in latest_signal_universe
        }
        
        print("✓ Cycle complete. Next check in 60 seconds...\n")
        time.sleep(60)

@app.on_event("startup")
def start_background_thread():
    print("[PLAYBOOK] Automatic entries and exits disabled.")

    def generate_initial_signals():
        """Generate first batch of signals immediately, then continue every 60 sec"""
        global latest_signals, latest_signal_universe, latest_volumes, top_volume_tickers
        print("\n📊 Generating initial signals...")
        for ticker in TICKERS:
            try:
                setup, data = evaluate_ticker(ticker)
                if setup is not None:
                    prob = 1.0 if setup.signal in {"BUY", "PUT"} else 0.0
                    latest_signal_universe[ticker] = {
                        "probability": float(prob),
                        "signal": setup.signal,
                        "setup": setup.as_dict(),
                    }
                    latest_volumes[ticker] = int(data.iloc[-1]["volume"]) if "volume" in data.columns else 0
                    print(f"  ✓ {ticker}: {setup.signal} ({setup.reason})")
                else:
                    latest_signal_universe[ticker] = {"probability": 0.0, "signal": "NO_DATA"}
            except Exception as e:
                print(f"  ❌ {ticker}: {e}")
                latest_signal_universe[ticker] = {"probability": 0.0, "signal": "ERROR"}

        sorted_by_volume = sorted(
            latest_volumes.items(),
            key=lambda item: item[1],
            reverse=True
        )
        top_volume_tickers = [ticker for ticker, _ in sorted_by_volume[:TOP_SIGNAL_COUNT]]
        latest_signals = {
            ticker: latest_signal_universe[ticker]
            for ticker in top_volume_tickers
            if ticker in latest_signal_universe
        }
        
        print("✓ Initial signals ready! Running loop in background...\n")
    
    # Initialize and load persistent history before trading loop starts
    init_history_db()
    init_trade_log_db()
    load_history_from_db()

    # Generate initial signals immediately
    generate_initial_signals()
    
    # Then start the background loop
    thread = threading.Thread(target=run_loop)
    thread.daemon = True
    thread.start()



@app.get("/signals")
def get_signals():
    """Get latest trading signals for top-volume tickers."""
    return latest_signals


@app.get("/zones/{ticker}")
def get_marked_zones(ticker: str):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Ticker is outside the trading universe")
    return {"ticker": ticker, "zones": load_marked_zones(ticker)}


@app.post("/zones/{ticker}")
def add_marked_zone(ticker: str, zone: MarkedZone):
    ticker = ticker.upper()
    if ticker not in TICKERS or zone.kind not in ZONE_KINDS or zone.high < zone.low:
        raise HTTPException(status_code=400, detail="Invalid ticker, zone kind, or bounds")
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO marked_zones (ticker, kind, low, high, created_at) VALUES (?, ?, ?, ?, ?)",
            (ticker, zone.kind, zone.low, zone.high, datetime.now(ET).isoformat()),
        )
        zone_id = cursor.lastrowid
    return {"id": zone_id, "ticker": ticker, **zone.model_dump()}


@app.delete("/zones/{ticker}/{zone_id}")
def delete_marked_zone(ticker: str, zone_id: int):
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Ticker is outside the trading universe")
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        deleted = conn.execute(
            "UPDATE marked_zones SET removed_at = ? WHERE id = ? AND ticker = ? AND removed_at IS NULL",
            (datetime.now(ET).isoformat(), zone_id, ticker),
        ).rowcount
    return {"deleted": bool(deleted)}

@app.get("/signals/universe")
def get_signal_universe(query: str = ""):
    """Get signal universe with top-volume set and optional ticker filter."""
    query_upper = query.strip().upper()
    universe_signals = latest_signal_universe
    if query_upper:
        universe_signals = {
            ticker: signal
            for ticker, signal in latest_signal_universe.items()
            if query_upper in ticker
        }

    return {
        "top_volume_tickers": top_volume_tickers,
        "signals": universe_signals,
        "available_tickers": sorted(latest_signal_universe.keys()),
    }

@app.get("/chart/{ticker}")
def get_chart(
    ticker: str,
    interval: str = Query(default="1m", pattern="^(1m|5m|15m|1h|4h|1d)$"),
    bars: int = Query(default=120, ge=30, le=500)
):
    """Get real broker OHLCV and playbook indicators for a universe ticker."""
    try:
        ticker = ticker.upper()
        if ticker not in TICKERS:
            raise HTTPException(status_code=400, detail="Ticker is outside the trading universe")
        minutes_lookup = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        lookback_minutes = bars * minutes_lookup[interval]
        data = get_latest_bars(ticker, minutes=lookback_minutes, interval=interval)
        if data is None or len(data) == 0:
            return {"ticker": ticker, "interval": interval, "bars": bars, "data": []}

        chart_points = get_chart_data(data)

        return {
            "ticker": ticker,
            "interval": interval,
            "bars": bars,
            "data": chart_points
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating chart data for {ticker} ({interval}): {e}")
        return {"ticker": ticker, "interval": interval, "bars": bars, "data": []}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/account/summary")
def get_account_summary():
    """Get account balance summary for dashboard cards."""
    account = get_account_info()
    if account is None:
        return {
            "equity": 0.0,
            "cash": 0.0,
            "buying_power": 0.0,
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "daily_pnl_percent": 0.0,
        }

    equity = float(account.equity)
    last_equity = float(account.last_equity) if hasattr(account, "last_equity") and account.last_equity is not None else equity
    daily_pnl = equity - last_equity
    daily_pnl_percent = (daily_pnl / last_equity * 100.0) if last_equity else 0.0

    return {
        "equity": equity,
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "portfolio_value": float(account.portfolio_value),
        "daily_pnl": daily_pnl,
        "daily_pnl_percent": daily_pnl_percent,
    }

@app.get("/history/{ticker}")
def get_history(ticker: str):
    """Get signal history for a ticker."""
    if ticker in signal_history:
        return {"ticker": ticker, "history": signal_history[ticker]}
    return {"ticker": ticker, "history": []}

@app.get("/history")
def get_all_history():
    """Get full signal history for all supported tickers."""
    all_history = []
    for ticker, records in signal_history.items():
        for record in records:
            all_history.append({
                "ticker": ticker,
                "timestamp": record["timestamp"],
                "probability": record["probability"],
                "signal": record["signal"]
            })

    # Newest first for dashboard readability
    all_history.sort(key=lambda item: item["timestamp"], reverse=True)
    return {
        "history": all_history,
        "total_records": len(all_history)
    }

@app.get("/metrics")
def get_metrics():
    """Count rule-confirmed ideas, not model probabilities."""
    metrics = {}
    for ticker in TICKERS:
        if ticker in signal_history and len(signal_history[ticker]) > 0:
            history = signal_history[ticker]
            metrics[ticker] = {
                "total_signals": len(history),
                "buy_signals": sum(s["signal"] == "BUY" for s in history),
                "short_signals": sum(s["signal"] == "PUT" for s in history),
            }
    return metrics

@app.get("/positions")
def get_positions_endpoint():
    """Report broker positions without inventing strategy stops or targets."""
    positions = [
        {**pos, "pnl_dollars": pos["pnl"]}
        for pos in get_account_positions() if pos["ticker"] in TICKERS
    ]
    return {"open_positions": positions, "total_open": len(positions)}

@app.get("/trades/log")
def get_trade_log(limit: int = 200):
    """
    Persisted completed trade / exit events with human-readable entry and exit reasons.
    Survives API restarts (stored in SQLite).
    """
    limit = max(1, min(int(limit), 2000))
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, ticker, entry_time, exit_time, entry_price, exit_price, qty_shares,
                   pnl_dollars, pnl_percent, entry_source, entry_probability, entry_threshold,
                   entry_reason_detail, exit_reason_code, exit_reason_detail, is_partial,
                   entry_order_id, created_at
            FROM trade_log
            ORDER BY datetime(exit_time) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "ticker": r["ticker"],
                "entry_time": r["entry_time"],
                "exit_time": r["exit_time"],
                "entry_price": r["entry_price"],
                "exit_price": r["exit_price"],
                "qty_shares": r["qty_shares"],
                "pnl_dollars": r["pnl_dollars"],
                "pnl_percent": r["pnl_percent"],
                "entry_source": r["entry_source"],
                "entry_probability": r["entry_probability"],
                "entry_threshold": r["entry_threshold"],
                "entry_reason_detail": r["entry_reason_detail"],
                "exit_reason_code": r["exit_reason_code"],
                "exit_reason_detail": r["exit_reason_detail"],
                "is_partial": bool(r["is_partial"]),
                "entry_order_id": r["entry_order_id"],
                "created_at": r["created_at"],
            }
        )
    return {"trades": out, "total_returned": len(out)}

@app.get("/orders")
def get_pending_orders():
    """Get all pending orders from Alpaca account."""
    try:
        orders = get_orders()
        
        orders_list = []
        for order in orders:
            # Determine order type
            if hasattr(order, 'stop_price') and order.stop_price:
                order_type = "STOP"
            elif order.side.lower() == "buy":
                order_type = "BUY"
            else:
                order_type = "SELL"
            
            orders_list.append({
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.upper(),
                "type": order_type,
                "status": order.status,
                "submitted_at": str(order.submitted_at) if hasattr(order, 'submitted_at') else None,
                "filled_qty": float(order.filled_qty) if hasattr(order, 'filled_qty') else 0,
                "limit_price": float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                "stop_price": float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
            })
        
        return {
            "orders": orders_list,
            "total_pending": len(orders_list)
        }
    except Exception as e:
        print(f"Error fetching orders: {e}")
        return {
            "orders": [],
            "total_pending": 0,
            "error": str(e)
        }

@app.post("/orders/place")
def place_order_endpoint(ticker: str, qty: int, order_type: str = "BUY"):
    """
    Manually place a market buy or sell order.
    
    Args:
        ticker: One of QQQ, SPY, TSLA, AAPL, NVDA, MSFT
        qty: Positive whole-share quantity
        order_type: "BUY" or "SELL"
    
    Returns:
        Order confirmation with order ID
    """
    try:
        # Validate inputs
        ticker = ticker.upper()
        if ticker not in TICKERS:
            return {"error": f"Invalid ticker. Must be one of: {TICKERS}", "success": False}
        
        if qty < 1:
            return {"error": "Quantity must be positive", "success": False}
        
        order_type = order_type.upper()
        if order_type not in ["BUY", "SELL"]:
            return {"error": "Order type must be BUY or SELL", "success": False}
        held = get_position(ticker)
        side = str(getattr(getattr(held, "side", None), "value", getattr(held, "side", ""))).lower()
        held_qty = abs(float(getattr(held, "qty", 0))) if held is not None else 0
        reducing = qty <= held_qty and (
            (order_type == "BUY" and side == "short")
            or (order_type == "SELL" and side == "long")
        )
        if not reducing:
            if not in_entry_window(datetime.now(ET)):
                return {"error": "New positions are limited to 09:30-11:00 ET; only position-reducing orders are allowed outside the window", "success": False}
            if held is not None and held_qty > 0 and side != ("long" if order_type == "BUY" else "short"):
                return {"error": "Order would reverse a position; close it first", "success": False}
            setup, _ = evaluate_ticker(ticker)
            required_signal = "BUY" if order_type == "BUY" else "PUT"
            if setup is None or setup.signal != required_signal:
                return {"error": "No currently confirmed playbook setup for this opening order", "success": False}
        
        # Place the order
        if order_type == "BUY":
            order = place_buy_order(ticker, qty)
        else:
            order = place_sell_order(ticker, qty)
        
        if order:
            return {
                "success": True,
                "order_id": order.id,
                "ticker": ticker,
                "qty": qty,
                "type": order_type,
                "status": order.status
            }
        else:
            return {"error": "Failed to place order", "success": False}
    
    except Exception as e:
        print(f"Error placing manual order: {e}")
        return {"error": str(e), "success": False}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Trading Bot Backend...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
