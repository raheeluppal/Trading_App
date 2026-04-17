from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time
import threading
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

from model import load_model, predict_signal
from data_feed import (
    get_latest_bars,
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
from position_manager import PositionManager

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
position_manager = PositionManager()  # Track open/closed positions

latest_signals = {}
latest_chart_data = {}  # Store chart data for each ticker
latest_prices = {}  # Store latest prices for position tracking
signal_history = {ticker: [] for ticker in ["SPY", "TSLA", "AMZN", "MSFT"]}  # Track signal history

TICKERS = ["SPY", "TSLA", "AMZN", "MSFT"]

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
                position_manager.open_position(ticker, entry, qty=int(qty))
                print(f"  ✓ {ticker}: {qty} shares @ ${entry:.2f} | Current: ${current:.2f} | P&L: {pnl_pct:.2f}% (${pnl:.2f})")
            else:
                # Update existing position price
                position = position_manager.open_positions[ticker]
                position.current_price = current
                print(f"  ✓ {ticker}: Updated price to ${current:.2f}")
    else:
        print("✓ No open positions on Alpaca account")

# -------------------------
# SIGNAL LOOP (every 60 sec)
# -------------------------
def calculate_position_size(account_balance, price, risk_percent=0.02):
    """
    Calculate position size based on account balance.
    Risk only risk_percent of account per trade (default 2%).
    
    Args:
        account_balance: Total account cash
        price: Stock price
        risk_percent: Percentage of account to risk (default 2%)
    
    Returns:
        Number of shares to buy
    """
    # Risk only 2% of account per position
    risk_amount = account_balance * risk_percent
    # Buy standard round lots (100 shares)
    position_value = max(100, int((risk_amount / price) / 100) * 100)
    return min(position_value // int(price), 100)  # Cap at 100 shares per position

def run_loop():
    global latest_signals, latest_chart_data, latest_prices

    print("\n🔄 Starting LIVE signal generation loop with REAL order placement...\n")
    
    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Checking signals and managing positions...\n")
        
        # Get account info for position sizing
        account = get_account_info()
        buying_power = float(account.buying_power) if account else 10000
        print(f"💰 Account Buying Power: ${buying_power:.2f}\n")
        
        prices_dict = {}
        atr_dict = {}
        
        for ticker in TICKERS:
            try:
                data = get_latest_bars(ticker)
                if data is not None and len(data) > 0:
                    # Get latest close price and ATR for position tracking
                    close_price = float(data.iloc[-1]['close'])
                    atr = calculate_atr(data)
                    
                    latest_prices[ticker] = close_price
                    prices_dict[ticker] = close_price
                    atr_dict[ticker] = atr if atr else close_price * 0.01
                    
                    # Build features for prediction
                    features = build_features(data)

                    # Generate prediction
                    prob = predict_signal(model, features)
                    is_buy_signal = prob > 0.50

                    # Store signal
                    latest_signals[ticker] = {
                        "probability": float(prob),
                        "signal": "BUY" if is_buy_signal else "WAIT"
                    }
                    
                    # ===== REAL ORDER PLACEMENT =====
                    # 1. If BUY signal and no open position, place real buy order
                    if is_buy_signal and ticker not in position_manager.open_positions:
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
                                order_id=buy_order.id
                            )
                            
                            # Place initial stop-loss order
                            stop_price = position.initial_stop_loss
                            stop_order = place_stop_order(ticker, qty, stop_price)
                            if stop_order:
                                position.stop_order_id = stop_order.id
                            
                            print(f"  ✅ {ticker}: OPENED REAL POSITION")
                            print(f"     └─ BUY ORDER: {qty} shares @ ${close_price:.2f} (Order: {buy_order.id[:8]}...)")
                            print(f"     └─ STOP LOSS: ${stop_price:.2f} (Order: {stop_order.id[:8] if stop_order else 'Failed'}...)\n")
                    
                    # Store chart data for frontend visualization
                    latest_chart_data[ticker] = get_chart_data(data)
                    
                    # Track signal history (keep last 100 signals)
                    signal_history[ticker].append({
                        "timestamp": str(datetime.now()),
                        "probability": float(prob),
                        "signal": "BUY" if is_buy_signal else "WAIT"
                    })
                    if len(signal_history[ticker]) > 100:
                        signal_history[ticker].pop(0)
                    
                    print(f"  {ticker}: {latest_signals[ticker]['signal']} (prob: {prob:.2%}) | Price: ${close_price:.2f} | ATR: ${atr:.2f}")
                    
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                latest_signals[ticker] = {
                    "probability": 0.0,
                    "signal": "ERROR"
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
                sell_order = place_sell_order(ticker, shares_to_sell)
                
                if sell_order:
                    # Cancel stop order if closing full position
                    if percent >= 1.0 and position.stop_order_id:
                        cancel_order(position.stop_order_id)
                    
                    # Track closure
                    position_manager.close_position(ticker, exit_reason, exit_price, percent)
                    
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
    def generate_initial_signals():
        """Generate first batch of signals immediately, then continue every 60 sec"""
        print("\n📊 Generating initial signals...")
        for ticker in TICKERS:
            try:
                data = get_latest_bars(ticker)
                if data is not None and len(data) > 0:
                    features = build_features(data)
                    prob = predict_signal(model, features)
                    latest_signals[ticker] = {
                        "probability": float(prob),
                        "signal": "BUY" if prob > 0.50 else "WAIT"
                    }
                    latest_chart_data[ticker] = get_chart_data(data)
                    print(f"  ✓ {ticker}: {latest_signals[ticker]['signal']} (prob: {prob:.2%})")
            except Exception as e:
                print(f"  ❌ {ticker}: {e}")
                latest_signals[ticker] = {"probability": 0.0, "signal": "ERROR"}
        
        print("✓ Initial signals ready! Running loop in background...\n")
    
    # First sync with real Alpaca positions
    sync_real_positions()
    
    # Generate initial signals immediately
    generate_initial_signals()
    
    # Then start the background loop
    thread = threading.Thread(target=run_loop)
    thread.daemon = True
    thread.start()

@app.get("/signals")
def get_signals():
    """Get latest trading signals for all tickers."""
    return latest_signals

@app.get("/chart/{ticker}")
def get_chart(ticker: str):
    """Get chart data (OHLCV + indicators) for a specific ticker."""
    if ticker in latest_chart_data:
        return {"ticker": ticker, "data": latest_chart_data[ticker]}
    return {"ticker": ticker, "data": []}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/history/{ticker}")
def get_history(ticker: str):
    """Get signal history for a ticker."""
    if ticker in signal_history:
        return {"ticker": ticker, "history": signal_history[ticker]}
    return {"ticker": ticker, "history": []}

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

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Trading Bot Backend...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
