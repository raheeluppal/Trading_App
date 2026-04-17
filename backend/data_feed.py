from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API credentials from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

# Initialize clients
data_client = None
trading_client = None

if API_KEY and API_SECRET:
    try:
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        # TradingClient doesn't take base_url parameter - it automatically uses paper trading
        trading_client = TradingClient(API_KEY, API_SECRET)
        print(f"✓ Alpaca data client initialized successfully")
        print(f"✓ Alpaca trading client initialized successfully")
        
        # Test connection
        account = trading_client.get_account()
        print(f"✓ Connected to Alpaca account | Buying Power: ${float(account.buying_power):.2f}")
    except Exception as e:
        print(f"Warning: Could not initialize Alpaca client: {e}")
        print("Will use mock data for development")
        data_client = None
        trading_client = None
else:
    print("⚠ No Alpaca credentials found in .env file")
    print("Using mock data for development")

# Keep reference to original client name for compatibility
client = data_client

INTERVAL_CONFIG = {
    "1m": {"timeframe": TimeFrame.Minute, "minutes_per_bar": 1},
    "5m": {"timeframe": TimeFrame(5, TimeFrame.Minute), "minutes_per_bar": 5},
    "15m": {"timeframe": TimeFrame(15, TimeFrame.Minute), "minutes_per_bar": 15},
    "1h": {"timeframe": TimeFrame.Hour, "minutes_per_bar": 60},
    "4h": {"timeframe": TimeFrame(4, TimeFrame.Hour), "minutes_per_bar": 240},
    "1d": {"timeframe": TimeFrame.Day, "minutes_per_bar": 1440},
}

def generate_mock_bars(ticker, bars=50, interval="1m"):
    """
    Generate mock OHLCV data for testing without real API.
    
    Args:
        ticker: Stock ticker symbol
        bars: Number of bars to generate
        interval: Chart interval (1m, 5m, 15m, 1h)
        
    Returns:
        DataFrame with simulated OHLCV data
    """
    interval_details = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["1m"])
    step_minutes = interval_details["minutes_per_bar"]
    now = datetime.now()
    timestamps = [now - timedelta(minutes=step_minutes * (bars - i)) for i in range(bars)]

    # Keep a stable shape per ticker, but let the stream evolve over time.
    # This avoids "frozen" charts when running without Alpaca credentials.
    base_seed = hash(ticker) % 2**32
    rng = np.random.default_rng(base_seed)
    time_bucket = int(now.timestamp() // (step_minutes * 60))
    
    base_price = {
        "SPY": 450,
        "TSLA": 250,
        "AMZN": 180,
        "MSFT": 380
    }.get(ticker, 100)
    
    returns = rng.normal(0.0001, 0.005, bars)
    prices = base_price * np.exp(np.cumsum(returns))
    drift = (time_bucket % 390) * 0.00025
    prices = prices * (1 + drift)
    
    df = pd.DataFrame({
        "open": prices * (1 + rng.uniform(-0.002, 0.002, bars)),
        "high": prices * (1 + rng.uniform(0, 0.003, bars)),
        "low": prices * (1 - rng.uniform(0, 0.003, bars)),
        "close": prices,
        "volume": rng.integers(1000000, 5000000, bars) + (time_bucket % 50) * 2500
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))
    
    return df

def get_latest_bars(ticker, minutes=50, interval="1m"):
    """
    Fetch latest bar data for a ticker.
    Uses Alpaca API if credentials are set, otherwise generates mock data.
    
    Args:
        ticker: Stock ticker symbol
        minutes: Number of minutes (legacy fallback)
        interval: Chart interval (1m, 5m, 15m, 1h)
        
    Returns:
        DataFrame with OHLCV data or None if error
    """
    if client is not None:
        try:
            interval_details = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["1m"])
            bars_to_fetch = max(int(minutes / interval_details["minutes_per_bar"]), 50)
            lookback_minutes = bars_to_fetch * interval_details["minutes_per_bar"]
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=interval_details["timeframe"],
                start=datetime.now() - timedelta(minutes=lookback_minutes),
                end=datetime.now()
            )

            bars = client.get_stock_bars(request).df
            return bars
        except Exception as e:
            print(f"Error fetching bars for {ticker}: {e}")
            print(f"Falling back to mock data for {ticker}")
            interval_details = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["1m"])
            bars_to_fetch = max(int(minutes / interval_details["minutes_per_bar"]), 50)
            return generate_mock_bars(ticker, bars=bars_to_fetch, interval=interval)
    else:
        # Use mock data for development
        try:
            interval_details = INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["1m"])
            bars_to_fetch = max(int(minutes / interval_details["minutes_per_bar"]), 50)
            return generate_mock_bars(ticker, bars=bars_to_fetch, interval=interval)
        except Exception as e:
            print(f"Error generating mock bars for {ticker}: {e}")
            return None

def get_realtime_price(ticker):
    """
    Fetch latest trade price for a ticker, if available.

    Returns:
        float price or None when unavailable.
    """
    if client is None:
        return None

    try:
        # Import lazily to avoid hard dependency mismatch issues.
        from alpaca.data.requests import StockLatestTradeRequest

        request = StockLatestTradeRequest(symbol_or_symbols=ticker)
        latest_trade = client.get_stock_latest_trade(request)

        trade_obj = latest_trade
        if isinstance(latest_trade, dict):
            trade_obj = latest_trade.get(ticker) or latest_trade.get(str(ticker))

        price = getattr(trade_obj, "price", None)
        return float(price) if price is not None else None
    except Exception:
        return None

def get_historical_bars_for_training(ticker, days=None):
    """
    Fetch historical daily bars for model training.
    Uses Alpaca API only (no synthetic fallback).
    
    Args:
        ticker: Stock ticker symbol
        days: Number of calendar days to fetch. If None, requests full available history.
        
    Returns:
        DataFrame with daily OHLCV data or None if unavailable/error
    """
    if client is None:
        print(f"Error: Alpaca data client unavailable. Cannot fetch historical bars for {ticker}.")
        return None

    end_date = datetime.now().date()
    # Ask for "all available" by using a far-back start date.
    start_date = datetime(2000, 1, 1).date() if days is None else (end_date - timedelta(days=int(days)))

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )

    last_error = None
    for attempt in range(1, 4):
        try:
            bars = client.get_stock_bars(request).df
            if bars is None or len(bars) == 0:
                print(f"Warning: no historical bars returned for {ticker}.")
                return None

            print(f"✓ Fetched {len(bars)} historical daily bars for {ticker} from Alpaca")
            return bars
        except Exception as e:
            last_error = e
            wait_s = min(8, 2 ** (attempt - 1))
            print(f"Error fetching historical data for {ticker} (attempt {attempt}/3): {e}")
            time.sleep(wait_s)

    print(f"Error fetching historical data for {ticker}: {last_error}")
    return None

# ==========================================
# ORDER PLACEMENT & POSITION MANAGEMENT
# ==========================================

def place_buy_order(ticker, qty):
    """
    Place a market buy order for a stock.
    
    Args:
        ticker: Stock ticker symbol
        qty: Quantity to buy (must be integer)
        
    Returns:
        Order object with order_id, or None if failed
    """
    if trading_client is None:
        print(f"⚠ Trading client not initialized. Cannot place order for {ticker}")
        return None
    
    try:
        # Ensure qty is integer
        qty = int(qty)
        if qty < 1:
            qty = 1
        
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_request)
        print(f"✓ BUY Order placed: {ticker} x{qty} | Order ID: {order.id}")
        return order
    except Exception as e:
        print(f"❌ Error placing buy order for {ticker}: {e}")
        return None

def place_sell_order(ticker, qty):
    """
    Place a market sell order for a stock.
    
    Args:
        ticker: Stock ticker symbol
        qty: Quantity to sell (must be integer)
        
    Returns:
        Order object with order_id, or None if failed
    """
    if trading_client is None:
        print(f"⚠ Trading client not initialized. Cannot place order for {ticker}")
        return None
    
    try:
        # Ensure qty is integer
        qty = int(qty)
        if qty < 1:
            qty = 1
        
        order_request = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        order = trading_client.submit_order(order_request)
        print(f"✓ SELL Order placed: {ticker} x{qty} | Order ID: {order.id}")
        return order
    except Exception as e:
        print(f"❌ Error placing sell order for {ticker}: {e}")
        return None

def place_stop_order(ticker, qty, stop_price):
    """
    Place a stop-loss order.
    
    Args:
        ticker: Stock ticker symbol
        qty: Quantity to sell on stop (must be integer)
        stop_price: Stop price level
        
    Returns:
        Order object with order_id, or None if failed
    """
    if trading_client is None:
        print(f"⚠ Trading client not initialized. Cannot place order for {ticker}")
        return None
    
    try:
        # Ensure qty is integer
        qty = int(qty)
        if qty < 1:
            qty = 1
        # Alpaca requires valid penny increments for this price range
        stop_price = round(float(stop_price), 2)
        
        order_request = StopOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC  # Good-til-cancelled
        )
        order = trading_client.submit_order(order_request)
        print(f"✓ STOP Order placed: {ticker} x{qty} @ ${stop_price:.2f} | Order ID: {order.id}")
        return order
    except Exception as e:
        print(f"❌ Error placing stop order for {ticker}: {e}")
        return None

def get_account_info():
    """
    Get account information (buying power, cash, etc).
    
    Returns:
        Account object with balance info
    """
    if trading_client is None:
        return None
    
    try:
        return trading_client.get_account()
    except Exception as e:
        print(f"❌ Error getting account info: {e}")
        return None

def get_positions():
    """
    Get all open positions from Alpaca account.
    
    Returns:
        List of position objects
    """
    if trading_client is None:
        return []
    
    try:
        positions = trading_client.get_all_positions()
        return positions
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        return []

def get_position(ticker):
    """
    Get a specific position for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Position object or None if not held
    """
    if trading_client is None:
        return None
    
    try:
        position = trading_client.get_open_position(ticker)
        return position
    except Exception as e:
        # Position not found is normal
        return None

def cancel_order(order_id):
    """
    Cancel an open order.
    
    Args:
        order_id: Order ID to cancel
        
    Returns:
        True if successful, False otherwise
    """
    if trading_client is None:
        return False
    
    try:
        trading_client.cancel_order_by_id(order_id)
        print(f"✓ Order cancelled: {order_id}")
        return True
    except Exception as e:
        print(f"❌ Error cancelling order {order_id}: {e}")
        return False

def get_orders():
    """
    Get all open orders.
    
    Returns:
        List of order objects
    """
    if trading_client is None:
        return []
    
    try:
        # Alpaca API: get_orders() without parameters returns open orders
        orders = trading_client.get_orders()
        return orders if orders else []
    except Exception as e:
        print(f"❌ Error getting orders: {e}")
        return []

def get_account_positions():
    """
    Get all open positions from Alpaca account as dictionaries.
    Useful for syncing real positions into position_manager.
    
    Returns:
        List of position dictionaries with: ticker, qty, entry_price, current_price, pnl
    """
    if trading_client is None:
        return []
    
    try:
        positions = trading_client.get_all_positions()
        if not positions:
            return []
        
        positions_list = []
        for pos in positions:
            # Get entry price from position attributes (varies by Alpaca API version)
            entry_price = float(pos.avg_fill_price) if hasattr(pos, 'avg_fill_price') else float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') else float(pos.current_price)
            
            positions_list.append({
                "ticker": pos.symbol,
                "qty": float(pos.qty),
                "entry_price": entry_price,
                "current_price": float(pos.current_price),
                "pnl": float(pos.unrealized_pl),
                "pnl_percent": float(pos.unrealized_plpc) * 100 if hasattr(pos, 'unrealized_plpc') else 0,
                "side": pos.side
            })
        
        return positions_list
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        return []

