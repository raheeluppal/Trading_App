from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import time
import threading
import numpy as np
import sqlite3
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# Load environment variables from .env file
load_dotenv()

from model import load_model, load_decision_threshold, predict_signal, reload_model_after_training
from data_feed import (
    get_latest_bars,
    get_realtime_price,
    get_account_positions,
    place_buy_order, 
    place_sell_order, 
    place_stop_order,
    get_account_info,
    get_positions,
    get_position,
    get_orders,
    cancel_order
)
from features import build_features, get_chart_data, calculate_atr
from position_manager import Position, PositionManager, format_exit_reason

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model()
SIGNAL_THRESHOLD = load_decision_threshold()
position_manager = PositionManager()  # Track open/closed positions
LIVE_TRADING_ENABLED = os.getenv("LIVE_TRADING_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Auto-retrain / hot-swap (see _auto_retrain_loop)
last_model_reload_at = None
last_model_reload_error = None
last_retrain_finished_at = None
retrain_in_progress = threading.Lock()
auto_retrain_running = False


def _run_train_and_swap():
    """Run train_model.train_model(), then reload global model + threshold from disk."""
    global model, SIGNAL_THRESHOLD
    global last_model_reload_at, last_model_reload_error, last_retrain_finished_at
    global auto_retrain_running

    if not retrain_in_progress.acquire(blocking=False):
        print("Auto-retrain: skipped (another run is in progress).")
        return

    auto_retrain_running = True
    try:
        import train_model as train_model_module

        print("\n═══ Retrain: started (full train_model.train_model) ═══")
        train_model_module.train_model()
        new_m, new_t = reload_model_after_training()
        model = new_m
        SIGNAL_THRESHOLD = new_t
        ts = datetime.now().isoformat()
        last_model_reload_at = ts
        last_retrain_finished_at = ts
        last_model_reload_error = None
        print(
            f"═══ Retrain: swapped model + threshold={new_t:.4f} at {ts} ═══\n"
        )
    except Exception as e:
        last_model_reload_error = repr(e)
        print(f"═══ Retrain FAILED (previous model unchanged): {e} ═══\n")
    finally:
        auto_retrain_running = False
        retrain_in_progress.release()


def _auto_retrain_scheduler_loop():
    """
    Background: wait AUTO_RETRAIN_FIRST_DELAY_HOURS (or interval), train+swap,
    then sleep AUTO_RETRAIN_INTERVAL_DAYS between cycles.
    Env: AUTO_RETRAIN_ENABLED=1, AUTO_RETRAIN_INTERVAL_DAYS=7, AUTO_RETRAIN_FIRST_DELAY_HOURS optional.
    """
    interval_days = float(os.getenv("AUTO_RETRAIN_INTERVAL_DAYS", "7"))
    if interval_days <= 0:
        print("Auto-retrain: AUTO_RETRAIN_INTERVAL_DAYS <= 0 — scheduler not started.")
        return

    interval_sec = interval_days * 86400.0
    first_hours = os.getenv("AUTO_RETRAIN_FIRST_DELAY_HOURS")
    if first_hours is None:
        first_sec = interval_sec
    else:
        first_sec = max(60.0, float(first_hours) * 3600.0)

    print(
        f"Auto-retrain: scheduler active — first cycle in {first_sec/3600:.1f}h, "
        f"then every {interval_days} day(s)."
    )
    time.sleep(first_sec)
    while True:
        _run_train_and_swap()
        time.sleep(interval_sec)


def _start_auto_retrain_if_enabled():
    enabled = os.getenv("AUTO_RETRAIN_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled:
        print("Auto-retrain: disabled (set AUTO_RETRAIN_ENABLED=1 to enable).")
        return
    t = threading.Thread(target=_auto_retrain_scheduler_loop, daemon=True, name="auto-retrain")
    t.start()

latest_signals = {}  # Current top-volume ticker signals only
latest_signal_universe = {}  # Signals for all tracked universe tickers
latest_chart_data = {}  # Store chart data for each ticker
latest_prices = {}  # Store latest prices for position tracking
latest_volumes = {}  # Last observed per-ticker volume
top_volume_tickers = []  # Computed top N by latest volume
signal_history = {}  # Track signal history
alert_rules = []
alert_events = []

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC",
    "SPY", "QQQ", "IWM", "XLF", "XLE", "BAC", "JPM", "PLTR", "COIN", "SOFI",
    "PFE", "F", "NIO", "DIS", "UBER"
]
TOP_SIGNAL_COUNT = 10
DB_PATH = Path(__file__).resolve().parent / "signal_history.db"
STARTING_EQUITY = 100000.0

for _ticker in TICKERS:
    signal_history[_ticker] = []

def init_history_db():
    """Initialize local SQLite database for signal history."""
    with sqlite3.connect(DB_PATH) as conn:
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
        conn.commit()

def save_signal_record(ticker: str, timestamp: str, probability: float, signal: str):
    """Persist one signal record to SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO signal_history (ticker, timestamp, probability, signal) VALUES (?, ?, ?, ?)",
            (ticker, timestamp, probability, signal)
        )
        conn.commit()


def init_trade_log_db():
    """Persistent completed-trade log (entry and exit narratives)."""
    with sqlite3.connect(DB_PATH) as conn:
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


def append_trade_log_row(position, exit_reason_code, exit_price, shares_closed, is_partial):
    """Persist one exit event (full or partial) to SQLite."""
    if not isinstance(position, Position):
        return
    entry_time = position.entry_time.isoformat() if position.entry_time else ""
    exit_time = (position.exit_time or datetime.now()).isoformat()
    pnl_dollars = (exit_price - position.entry_price) * float(shares_closed)
    pnl_pct = (
        (exit_price / position.entry_price - 1.0) * 100.0
        if position.entry_price
        else 0.0
    )
    entry_detail = position.get_entry_reason_detail()
    exit_detail = format_exit_reason(exit_reason_code, position)
    created_at = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO trade_log (
                ticker, entry_time, exit_time, entry_price, exit_price, qty_shares,
                pnl_dollars, pnl_percent, entry_source, entry_probability, entry_threshold,
                entry_reason_detail, exit_reason_code, exit_reason_detail, is_partial,
                entry_order_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.ticker,
                entry_time,
                exit_time,
                float(position.entry_price),
                float(exit_price),
                int(shares_closed),
                round(pnl_dollars, 4),
                round(pnl_pct, 4),
                position.entry_source,
                position.entry_probability,
                position.entry_threshold,
                entry_detail,
                exit_reason_code,
                exit_detail,
                1 if is_partial else 0,
                str(position.order_id) if position.order_id else None,
                created_at,
            ),
        )
        conn.commit()


def load_history_from_db():
    """Load all historical signal records from SQLite into memory cache."""
    for ticker in TICKERS:
        signal_history[ticker] = []

    with sqlite3.connect(DB_PATH) as conn:
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

# ===========================
# POSITION SYNCHRONIZATION
# ===========================
def sync_real_positions():
    """
    Load all real Alpaca positions into position_manager.
    Called on startup to ensure dashboard shows real positions.
    """
    print("\n🔄 Syncing with Alpaca account positions...")
    real_positions = get_account_positions()
    
    if real_positions:
        print(f"✓ Found {len(real_positions)} open position(s) on Alpaca:")
        for pos in real_positions:
            ticker = pos["ticker"]
            qty = pos["qty"]
            entry = pos["entry_price"]
            current = pos["current_price"]
            pnl = pos["pnl"]
            pnl_pct = pos["pnl_percent"]
            
            # Add to position_manager if not already tracked
            if ticker not in position_manager.open_positions:
                position_manager.open_position(
                    ticker, entry, qty=int(qty), entry_source="sync"
                )
                print(f"  ✓ {ticker}: {qty} shares @ ${entry:.2f} | Current: ${current:.2f} | P&L: {pnl_pct:.2f}% (${pnl:.2f})")
            else:
                # Update existing position price
                position = position_manager.open_positions[ticker]
                position.current_price = current
                print(f"  ✓ {ticker}: Updated price to ${current:.2f}")
    else:
        print("✓ No open positions on Alpaca account")


def close_all_positions_on_startup():
    """Close any pre-existing broker positions before the live loop starts."""
    positions = get_account_positions()
    if not positions:
        print("[OK] No pre-existing positions to close on startup.")
        return

    print(f"[WARN] Closing {len(positions)} pre-existing position(s) on startup.")
    for pos in positions:
        try:
            ticker = str(pos.get("ticker", "")).upper()
            qty = int(float(pos.get("qty", 0)))
            if not ticker or qty <= 0:
                continue
            print(f"[WARN] Closing pre-existing position: {ticker} x{qty}")
            place_sell_order(ticker, qty)
        except Exception as e:
            print(f"[ERROR] Failed to close startup position {pos}: {e}")

# -------------------------
# SIGNAL LOOP (every 60 sec)
# -------------------------
def calculate_position_size(account_balance, price, risk_percent=0.001):
    """
    Calculate position size based on account balance.
    Test mode sizing: risk only risk_percent per trade (default 0.1%).
    
    Args:
        account_balance: Total account cash
        price: Stock price
        risk_percent: Percentage of account to risk (default 0.1%)
    
    Returns:
        Number of shares to buy
    """
    # Test mode: very small sizing until signal quality is validated.
    risk_amount = account_balance * risk_percent
    position_value = max(1, int((risk_amount / max(price, 1e-9)) / 1) * 1)
    shares = max(1, position_value // max(int(price), 1))
    return min(shares, 5)  # Hard cap for paper-test safety

def run_loop():
    global latest_signals, latest_signal_universe, latest_chart_data, latest_prices, latest_volumes, top_volume_tickers

    print("\n🔄 Starting LIVE signal generation loop with REAL order placement...\n")
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking signals and managing positions...\n")
        
        # Get account info for position sizing
        account = get_account_info()
        buying_power = float(account.buying_power) if account else 10000
        print(f"💰 Account Buying Power: ${buying_power:.2f}\n")
        
        prices_dict = {}
        atr_dict = {}
        try:
            spy_ctx = get_latest_bars("SPY")
        except Exception:
            spy_ctx = None

        for ticker in TICKERS:
            try:
                data = get_latest_bars(ticker)
                if data is not None and len(data) > 0:
                    # Get latest close price and ATR for position tracking
                    close_price = float(data.iloc[-1]['close'])
                    atr = calculate_atr(data)
                    
                    latest_prices[ticker] = close_price
                    latest_volumes[ticker] = int(data.iloc[-1]["volume"]) if "volume" in data.columns else 0
                    prices_dict[ticker] = close_price
                    atr_dict[ticker] = atr if atr else close_price * 0.01
                    
                    # Build features for prediction (SPY series = market for cross-sectional mkt_ret_*)
                    use_spy = data if ticker == "SPY" else spy_ctx
                    features = build_features(data, spy_bars=use_spy)

                    # Generate prediction
                    prob = predict_signal(model, features, ticker=ticker)
                    is_buy_signal = prob >= SIGNAL_THRESHOLD
                    print("\n[Signal Debug Info]")
                    print(f"| Probability: {float(prob):.6f}")
                    print(f"| Threshold: {float(SIGNAL_THRESHOLD):.6f}")
                    print(f"| Close price: ${close_price:.2f}")
                    print("| Features sample:")
                    print(f"|  |- RSI: {float(features.get('rsi', 0.0)):.2f}")
                    print(f"|  |- MACD: {float(features.get('macd_diff', 0.0)):.4f}")
                    print(f"|  |- Volume ratio: {float(features.get('volume_ratio', 0.0)):.2f}")
                    print(f"|  `- Market ret 1: {float(features.get('mkt_ret_1', 0.0)):.4f}")
                    print(f"`- is_buy_signal: {is_buy_signal}")
                    rsi_value = float(features.get("rsi", 0.0))

                    # Store signal
                    latest_signal_universe[ticker] = {
                        "probability": float(prob),
                        "signal": "BUY" if is_buy_signal else "WAIT"
                    }

                    # Evaluate active alert rules against latest market data
                    now_iso = datetime.now().isoformat()
                    for rule in alert_rules:
                        if not rule.get("enabled", True):
                            continue
                        if rule.get("ticker") != ticker:
                            continue

                        trigger_value = None
                        if rule.get("metric") == "price":
                            trigger_value = close_price
                        elif rule.get("metric") == "rsi":
                            trigger_value = rsi_value
                        elif rule.get("metric") == "probability":
                            trigger_value = float(prob) * 100

                        if trigger_value is None:
                            continue

                        threshold = float(rule.get("threshold", 0))
                        condition = rule.get("condition", "above")
                        cooldown_seconds = int(rule.get("cooldown_seconds", 300))
                        last_triggered_at = rule.get("last_triggered_at")

                        cooldown_ok = True
                        if last_triggered_at:
                            try:
                                elapsed = (datetime.now() - datetime.fromisoformat(last_triggered_at)).total_seconds()
                                cooldown_ok = elapsed >= cooldown_seconds
                            except Exception:
                                cooldown_ok = True

                        if not cooldown_ok:
                            continue

                        is_triggered = (condition == "above" and trigger_value > threshold) or (
                            condition == "below" and trigger_value < threshold
                        )
                        if is_triggered:
                            rule["last_triggered_at"] = now_iso
                            event = {
                                "id": str(uuid4()),
                                "rule_id": rule["id"],
                                "ticker": ticker,
                                "metric": rule["metric"],
                                "condition": condition,
                                "threshold": threshold,
                                "value": round(float(trigger_value), 4),
                                "timestamp": now_iso,
                                "message": f"{ticker} {rule['metric']} is {trigger_value:.2f}, {condition} {threshold:.2f}",
                            }
                            alert_events.insert(0, event)
                            if len(alert_events) > 500:
                                alert_events.pop()
                    
                    # ===== REAL ORDER PLACEMENT =====
                    # 1. If BUY signal and no open position, place real buy order
                    if is_buy_signal and ticker not in position_manager.open_positions:
                        if not LIVE_TRADING_ENABLED:
                            print(
                                f"[SAFE MODE] LIVE_TRADING_ENABLED=0 -> "
                                f"skipping BUY for {ticker} at prob={float(prob):.4f}"
                            )
                            continue
                        # Calculate position size
                        qty = calculate_position_size(buying_power, close_price)
                        
                        # Place market buy order
                        buy_order = place_buy_order(ticker, qty)
                        
                        if buy_order:
                            # Track position locally
                            position = position_manager.open_position(
                                ticker,
                                close_price,
                                atr,
                                qty=qty,
                                order_id=buy_order.id,
                                entry_probability=float(prob),
                                entry_threshold=float(SIGNAL_THRESHOLD),
                                entry_source="model",
                            )
                            
                            # Place initial stop-loss order
                            stop_price = position.initial_stop_loss
                            stop_order = place_stop_order(ticker, qty, stop_price)
                            if stop_order:
                                position.stop_order_id = stop_order.id
                            
                            buy_order_id = str(buy_order.id)
                            stop_order_id = str(stop_order.id) if stop_order else "Failed"
                            print(f"  ✅ {ticker}: OPENED REAL POSITION")
                            print(f"     └─ BUY ORDER: {qty} shares @ ${close_price:.2f} (Order: {buy_order_id[:8]}...)")
                            print(f"     └─ STOP LOSS: ${stop_price:.2f} (Order: {stop_order_id[:8] if stop_order else stop_order_id}...)\n")
                    
                    # Store chart data for frontend visualization
                    latest_chart_data[ticker] = get_chart_data(data)
                    
                    # Track signal history for dashboard records
                    record_timestamp = datetime.now().isoformat()
                    record_signal = "BUY" if is_buy_signal else "WAIT"
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
                    
                    print(f"  {ticker}: {latest_signal_universe[ticker]['signal']} (prob: {prob:.2%}) | Price: ${close_price:.2f} | ATR: ${atr:.2f}")
                    
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
        
        # 2. Check for exit signals on all open positions
        positions_to_close = position_manager.update_prices(prices_dict, atr_dict)
        for close_info in positions_to_close:
            ticker = close_info['ticker']
            percent = close_info['percent_exiting']
            position = close_info['position']
            exit_reason = close_info['exit_reason']
            exit_price = close_info['exit_price']
            
            # Calculate shares to sell
            shares_to_sell = int(position.qty * percent)
            
            # Place real sell order
            if shares_to_sell > 0:
                if not LIVE_TRADING_ENABLED:
                    print(
                        f"[SAFE MODE] LIVE_TRADING_ENABLED=0 -> skipping SELL "
                        f"for {ticker} ({shares_to_sell} shares, reason={exit_reason})"
                    )
                    continue
                sell_order = place_sell_order(ticker, shares_to_sell)
                
                if sell_order:
                    # Cancel stop order if closing full position
                    if percent >= 1.0 and position.stop_order_id:
                        cancel_order(position.stop_order_id)
                    
                    # Track closure
                    position_manager.close_position(ticker, exit_reason, exit_price, percent)
                    append_trade_log_row(
                        position,
                        exit_reason,
                        exit_price,
                        shares_to_sell,
                        is_partial=(percent < 1.0),
                    )

                    pnl_percent = position.get_pnl()
                    pnl_dollars = position.get_pnl_dollars() * shares_to_sell
                    
                    if percent < 1.0:
                        print(f"  ✅ {ticker}: PARTIAL CLOSE ({percent*100:.0f}%)")
                        print(f"     └─ REASON: {exit_reason}")
                        print(f"     └─ SELL ORDER: {shares_to_sell} shares @ ${exit_price:.2f}")
                        print(f"     └─ P&L: {pnl_percent:.2%} (${pnl_dollars:+.2f})\n")
                    else:
                        print(f"  ✅ {ticker}: CLOSED POSITION")
                        print(f"     └─ REASON: {exit_reason}")
                        print(f"     └─ SELL ORDER: {shares_to_sell} shares @ ${exit_price:.2f}")
                        print(f"     └─ P&L: {pnl_percent:.2%} (${pnl_dollars:+.2f})\n")

        print("✓ Cycle complete. Next check in 60 seconds...\n")
        time.sleep(60)

@app.on_event("startup")
def start_background_thread():
    print(
        f"[SAFE MODE] Live trading enabled: {LIVE_TRADING_ENABLED} "
        f"(set LIVE_TRADING_ENABLED=1 to allow order placement)"
    )

    def generate_initial_signals():
        """Generate first batch of signals immediately, then continue every 60 sec"""
        global latest_signals, latest_signal_universe, latest_volumes, top_volume_tickers
        print("\n📊 Generating initial signals...")
        try:
            spy_ctx0 = get_latest_bars("SPY")
        except Exception:
            spy_ctx0 = None
        for ticker in TICKERS:
            try:
                data = get_latest_bars(ticker)
                if data is not None and len(data) > 0:
                    use_spy = data if ticker == "SPY" else spy_ctx0
                    features = build_features(data, spy_bars=use_spy)
                    prob = predict_signal(model, features, ticker=ticker)
                    latest_signal_universe[ticker] = {
                        "probability": float(prob),
                        "signal": "BUY" if prob >= SIGNAL_THRESHOLD else "WAIT"
                    }
                    latest_volumes[ticker] = int(data.iloc[-1]["volume"]) if "volume" in data.columns else 0
                    latest_chart_data[ticker] = get_chart_data(data)
                    print(f"  ✓ {ticker}: {latest_signal_universe[ticker]['signal']} (prob: {prob:.2%})")
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

    # Hard safety: flatten broker positions before enabling any new signals/orders.
    close_all_positions_on_startup()

    # First sync with real Alpaca positions
    sync_real_positions()
    
    # Generate initial signals immediately
    generate_initial_signals()
    
    # Then start the background loop
    thread = threading.Thread(target=run_loop)
    thread.daemon = True
    thread.start()

    _start_auto_retrain_if_enabled()


@app.get("/model/status")
def model_status():
    """Training artifact reload status (manual + automatic)."""
    return {
        "signal_threshold": float(SIGNAL_THRESHOLD),
        "last_model_reload_at": last_model_reload_at,
        "last_retrain_finished_at": last_retrain_finished_at,
        "last_error": last_model_reload_error,
        "auto_retrain_running": auto_retrain_running,
        "auto_retrain_enabled_env": os.getenv("AUTO_RETRAIN_ENABLED", "0"),
        "auto_retrain_interval_days": os.getenv("AUTO_RETRAIN_INTERVAL_DAYS", "7"),
    }


@app.post("/internal/retrain")
def trigger_retrain_now(token: str = Query(..., description="Must match RETRAIN_TRIGGER_SECRET")):
    """Kick off train_model + model swap in a background thread (optional manual trigger)."""
    secret = os.getenv("RETRAIN_TRIGGER_SECRET", "").strip()
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="RETRAIN_TRIGGER_SECRET is not set in the environment.",
        )
    if token != secret:
        raise HTTPException(status_code=403, detail="Invalid token")

    threading.Thread(target=_run_train_and_swap, daemon=True, name="manual-retrain").start()
    return {"status": "started", "message": "Retrain running in background; check /model/status"}


@app.get("/signals")
def get_signals():
    """Get latest trading signals for top-volume tickers."""
    return latest_signals

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
    """Get chart data (OHLCV + indicators) for a specific ticker and interval."""
    try:
        minutes_lookup = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        lookback_minutes = bars * minutes_lookup[interval]
        data = get_latest_bars(ticker, minutes=lookback_minutes, interval=interval)
        if data is None or len(data) == 0:
            return {"ticker": ticker, "interval": interval, "bars": bars, "data": []}

        chart_points = get_chart_data(data)

        # For 1m charts, apply latest trade as a live "forming candle" update.
        # This makes the chart move between minute closes.
        if interval == "1m" and chart_points:
            live_price = get_realtime_price(ticker)
            if live_price is not None:
                last = chart_points[-1]
                last_open = float(last["open"])
                last["close"] = float(live_price)
                last["high"] = max(float(last["high"]), float(live_price), last_open)
                last["low"] = min(float(last["low"]), float(live_price), last_open)

        return {
            "ticker": ticker,
            "interval": interval,
            "bars": bars,
            "data": chart_points
        }
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
            "starting_equity": STARTING_EQUITY,
            "account_pnl": 0.0,
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
        "starting_equity": STARTING_EQUITY,
        "account_pnl": equity - STARTING_EQUITY,
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
    """Get performance metrics across all tickers."""
    metrics = {}
    for ticker in TICKERS:
        if ticker in signal_history and len(signal_history[ticker]) > 0:
            history = signal_history[ticker]
            buy_signals = sum(1 for s in history if s["signal"] == "BUY")
            avg_prob = np.mean([s["probability"] for s in history])
            max_prob = np.max([s["probability"] for s in history])
            min_prob = np.min([s["probability"] for s in history])
            
            metrics[ticker] = {
                "total_signals": len(history),
                "buy_signals": buy_signals,
                "avg_probability": round(float(avg_prob), 2),
                "max_probability": round(float(max_prob), 2),
                "min_probability": round(float(min_prob), 2),
            }
    return metrics

@app.get("/positions")
def get_positions_endpoint():
    """Get all open trading positions with exit targets (both real and tracked)."""
    # Get locally tracked positions
    tracked_positions = position_manager.get_open_positions()
    
    # Get real Alpaca positions
    real_positions = get_account_positions()
    
    # Merge them (real positions take precedence)
    all_positions = tracked_positions.copy()
    
    # Update with real position data
    for real_pos in real_positions:
        ticker = real_pos["ticker"]
        # Find if we have a tracked position for this ticker
        tracked = next((p for p in all_positions if p["ticker"] == ticker), None)
        
        if tracked:
            # Update tracked position with real data
            tracked["current_price"] = real_pos["current_price"]
            tracked["pnl_percent"] = real_pos["pnl_percent"]
            tracked["pnl_dollars"] = real_pos["pnl"]
        else:
            # Add new real position not in tracked
            all_positions.append({
                "ticker": real_pos["ticker"],
                "status": "OPEN",
                "entry_price": real_pos["entry_price"],
                "current_price": real_pos["current_price"],
                "highest_price": real_pos["current_price"],
                "entry_time": datetime.now().isoformat(),
                "hold_time_minutes": 0,
                "pnl_percent": real_pos["pnl_percent"],
                "pnl_dollars": real_pos["pnl"],
                "risk_dollars": real_pos["entry_price"] * 0.01,
                "initial_stop": real_pos["entry_price"] * 0.99,
                "current_stop": real_pos["entry_price"] * 0.99,
                "profit_target": real_pos["entry_price"] * 1.02,
                "stop_loss": -1.0,
                "atr": real_pos["entry_price"] * 0.01,
                "stop_distance_percent": ((real_pos["current_price"] - (real_pos["entry_price"] * 0.99)) / real_pos["current_price"] * 100),
                "breakeven_activated": False,
                "trailing_stop_active": False,
                "profit_targets": [
                    {"level_percent": 1.0, "size_percent": 25, "hit": False, "remaining_size": 100},
                    {"level_percent": 2.0, "size_percent": 35, "hit": False, "remaining_size": 100},
                    {"level_percent": 3.0, "size_percent": 40, "hit": False, "remaining_size": 100}
                ],
                "partial_exits": 0,
                "total_exited_percent": 0,
                "max_hold_minutes": 240,
                "exit_reason": None,
                "exit_price": None,
                "exit_time": None
            })
    
    return {
        "open_positions": all_positions,
        "total_open": len(all_positions)
    }

@app.get("/positions/closed")
def get_closed_positions():
    """Get all closed positions and trading history."""
    closed_positions = position_manager.get_closed_positions()
    return {
        "closed_positions": closed_positions,
        "total_closed": len(closed_positions)
    }


@app.get("/trades/log")
def get_trade_log(limit: int = 200):
    """
    Persisted completed trade / exit events with human-readable entry and exit reasons.
    Survives API restarts (stored in SQLite).
    """
    limit = max(1, min(int(limit), 2000))
    with sqlite3.connect(DB_PATH) as conn:
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

@app.get("/positions/stats")
def get_position_stats():
    """Get trading statistics (win rate, P&L, etc)."""
    stats = position_manager.get_statistics()
    return {
        "statistics": stats,
        "open_positions": len(position_manager.open_positions),
        "closed_positions": len(position_manager.closed_positions)
    }

@app.get("/positions/dashboard")
def get_positions_dashboard():
    """Get comprehensive position and trading dashboard."""
    # Get locally tracked positions
    open_pos = position_manager.get_open_positions()
    
    # Get real Alpaca positions
    real_positions = get_account_positions()
    
    # Merge them (real positions take precedence)
    all_open = open_pos.copy()
    for real_pos in real_positions:
        ticker = real_pos["ticker"]
        tracked = next((p for p in all_open if p["ticker"] == ticker), None)
        if tracked:
            tracked["current_price"] = real_pos["current_price"]
            tracked["pnl_percent"] = real_pos["pnl_percent"]
            tracked["pnl_dollars"] = real_pos["pnl"]
        else:
            all_open.append({
                "ticker": ticker,
                "status": "OPEN",
                "entry_price": real_pos["entry_price"],
                "current_price": real_pos["current_price"],
                "highest_price": real_pos["current_price"],
                "entry_time": datetime.now().isoformat(),
                "hold_time_minutes": 0,
                "pnl_percent": real_pos["pnl_percent"],
                "pnl_dollars": real_pos["pnl"],
                "risk_dollars": 0,
                "initial_stop": 0,
                "current_stop": 0,
                "atr": 0,
                "stop_distance_percent": 0,
                "breakeven_activated": False,
                "trailing_stop_active": False,
                "profit_targets": [],
                "partial_exits": 0,
                "total_exited_percent": 0,
                "max_hold_minutes": 0,
                "exit_reason": None,
                "exit_price": None,
                "exit_time": None
            })
    
    closed_pos = position_manager.get_closed_positions()
    stats = position_manager.get_statistics()
    
    # Calculate total P&L across open positions
    total_open_pnl = sum(p.get("pnl_dollars", 0) for p in all_open)
    
    return {
        "open_positions": all_open,
        "closed_positions": closed_pos,
        "statistics": stats,
        "summary": {
            "total_open": len(all_open),
            "total_closed": len(closed_pos),
            "total_trades": stats["total_trades"],
            "win_rate": stats["win_rate"],
            "total_pnl": stats["total_pnl"] + total_open_pnl
        }
    }

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
        ticker: Stock ticker (SPY, TSLA, AMZN, MSFT)
        qty: Number of shares (1-100)
        order_type: "BUY" or "SELL"
    
    Returns:
        Order confirmation with order ID
    """
    try:
        # Validate inputs
        if ticker not in TICKERS:
            return {"error": f"Invalid ticker. Must be one of: {TICKERS}", "success": False}
        
        if qty < 1 or qty > 100:
            return {"error": "Quantity must be between 1 and 100 shares", "success": False}
        
        order_type = order_type.upper()
        if order_type not in ["BUY", "SELL"]:
            return {"error": "Order type must be BUY or SELL", "success": False}
        
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

@app.get("/alerts/rules")
def get_alert_rules():
    """Get all configured alert rules."""
    return {"rules": alert_rules}

@app.post("/alerts/rules")
def create_alert_rule(
    ticker: str,
    metric: str = "price",
    condition: str = "above",
    threshold: float = 0.0,
    cooldown_seconds: int = 300
):
    """Create alert rule for price, RSI, or model probability."""
    if ticker not in TICKERS:
        return {"success": False, "error": f"Invalid ticker. Must be one of: {TICKERS}"}
    if metric not in ["price", "rsi", "probability"]:
        return {"success": False, "error": "metric must be price, rsi, or probability"}
    if condition not in ["above", "below"]:
        return {"success": False, "error": "condition must be above or below"}

    rule = {
        "id": str(uuid4()),
        "ticker": ticker,
        "metric": metric,
        "condition": condition,
        "threshold": float(threshold),
        "cooldown_seconds": int(cooldown_seconds),
        "enabled": True,
        "last_triggered_at": None,
        "created_at": datetime.now().isoformat(),
    }
    alert_rules.append(rule)
    return {"success": True, "rule": rule}

@app.delete("/alerts/rules/{rule_id}")
def delete_alert_rule(rule_id: str):
    """Delete an alert rule by ID."""
    global alert_rules
    existing_len = len(alert_rules)
    alert_rules = [rule for rule in alert_rules if rule["id"] != rule_id]
    return {"success": len(alert_rules) < existing_len}

@app.get("/alerts/events")
def get_alert_events(limit: int = 50):
    """Get latest alert trigger events."""
    return {"events": alert_events[:limit], "total": len(alert_events)}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Trading Bot Backend...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
