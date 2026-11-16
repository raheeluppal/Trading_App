from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta
import os
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API credentials from environment variables
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
STOCK_DATA_FEED = DataFeed.SIP if os.getenv("ALPACA_DATA_FEED", "iex").lower() == "sip" else DataFeed.IEX

# Initialize clients
data_client = None
trading_client = None

if API_KEY and API_SECRET:
    try:
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        print("Alpaca data client initialized")
    except Exception as e:
        print(f"Warning: Could not initialize Alpaca data client: {e}")
        data_client = None
    try:
        # Brokerage availability must not determine historical-data availability.
        trading_client = TradingClient(API_KEY, API_SECRET)
        print("Alpaca trading client initialized")
    except Exception as e:
        print(f"Warning: Could not initialize Alpaca trading client: {e}")
        trading_client = None
else:
    print("No Alpaca credentials found in .env file")
    print("Market data unavailable until Alpaca credentials are configured")

client = data_client

INTERVAL_CONFIG = {
    "1m": {"timeframe": TimeFrame.Minute, "minutes_per_bar": 1},
    "5m": {"timeframe": TimeFrame(5, TimeFrame.Minute), "minutes_per_bar": 5},
    "15m": {"timeframe": TimeFrame(15, TimeFrame.Minute), "minutes_per_bar": 15},
    "1h": {"timeframe": TimeFrame.Hour, "minutes_per_bar": 60},
    "4h": {"timeframe": TimeFrame(4, TimeFrame.Hour), "minutes_per_bar": 240},
    "1d": {"timeframe": TimeFrame.Day, "minutes_per_bar": 1440},
}

def get_latest_bars(ticker, minutes=50, interval="1m"):
    """
    Fetch latest bar data for a ticker.
    Uses Alpaca data only; never substitutes synthetic prices.
    
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
            now = datetime.now()

            # For intraday bars, minute-based lookback can fail at/near 9:30 because there may be
            # too few same-day bars. Pull a few calendar days and trim to the latest N bars.
            if interval_details["minutes_per_bar"] < 1440:
                lookback_days = max(
                    2,
                    int(math.ceil(lookback_minutes / (60.0 * 6.5))) + 1,
                )
                start_dt = now - timedelta(days=lookback_days)
            else:
                start_dt = now - timedelta(minutes=lookback_minutes)

            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=interval_details["timeframe"],
                start=start_dt,
                end=now,
                feed=STOCK_DATA_FEED,
            )

            bars = client.get_stock_bars(request).df
            if bars is None or len(bars) == 0:
                return bars

            out = bars.copy().sort_index()
            if len(out) > bars_to_fetch:
                out = out.tail(bars_to_fetch)
            return out
        except Exception as e:
            print(f"Error fetching bars for {ticker}: {e}")
            return None
    else:
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
    Used only for displaying actual broker holdings.
    
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
                "side": str(getattr(pos.side, "value", pos.side))
            })
        
        return positions_list
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        return []

