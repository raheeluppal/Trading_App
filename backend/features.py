import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with 'close' column
        window: Period for moving average (default 20)
        num_std: Number of standard deviations (default 2)
        
    Returns:
        DataFrame with bb_upper, bb_middle, bb_lower, bb_position, bb_width
    """
    close = df["close"]
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
    # BB Position: 0 = at lower band, 1 = at upper band
    bb_position = (close - lower_band) / (upper_band - lower_band)
    bb_position = bb_position.clip(0, 1)
    
    # BB Width as % of price
    bb_width = (upper_band - lower_band) / close * 100
    
    return {
        "bb_upper": upper_band,
        "bb_middle": sma,
        "bb_lower": lower_band,
        "bb_position": bb_position,
        "bb_width": bb_width
    }

def build_features(df):
    """
    Build enhanced technical indicators from OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with 10 computed features for ML model
    """
    df = df.copy()

    # === Original Features ===
    # Calculate Bollinger Bands
    bb = calculate_bollinger_bands(df)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"] = bb["bb_lower"]
    df["bb_position"] = bb["bb_position"]
    df["bb_width"] = bb["bb_width"]
    
    # Calculate RSI
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    
    # Calculate MACD
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    # Calculate Volume Ratio
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

    # === New Enhanced Features ===
    # 1. Momentum: Price change over 20 periods
    df["momentum"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20)
    df["momentum"] = df["momentum"].fillna(0)
    
    # 2. Volume Trend: Volume change
    df["volume_change"] = (df["volume"] - df["volume"].shift(5)) / (df["volume"].shift(5) + 1e-10)
    df["volume_change"] = df["volume_change"].fillna(0)
    
    # 3. Moving Average Signal: Fast MA (10) > Slow MA (30)
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_30"] = df["close"].rolling(window=30).mean()
    df["ma_signal"] = (df["sma_10"] > df["sma_30"]).astype(float)
    df["ma_signal"] = df["ma_signal"].fillna(0)
    
    # 4. ATR (Volatility): Average True Range
    try:
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        df["atr_normalized"] = df["atr"] / df["close"]
    except:
        df["atr_normalized"] = 0
    df["atr_normalized"] = df["atr_normalized"].fillna(0)
    
    # 5. ROC (Rate of Change): 12-period momentum
    try:
        df["roc"] = ROCIndicator(df["close"], window=12).roc()
    except:
        df["roc"] = 0
    df["roc"] = df["roc"].fillna(0)

    # Get the latest values (drop NaN)
    latest = df.dropna().iloc[-1] if len(df.dropna()) > 0 else df.iloc[-1]

    return {
        "bb_position": float(latest["bb_position"]),
        "bb_width": float(latest["bb_width"]),
        "rsi": float(latest["rsi"]),
        "macd_diff": float(latest["macd_diff"]),
        "volume_ratio": float(latest["volume_ratio"]),
        "momentum": float(latest["momentum"]),
        "volume_change": float(latest["volume_change"]),
        "ma_signal": float(latest["ma_signal"]),
        "atr_normalized": float(latest["atr_normalized"]),
        "roc": float(latest["roc"])
    }

def get_chart_data(df):
    """
    Get full chart data with all indicators for frontend visualization.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with chart data
    """
    df = df.copy()
    
    # Calculate all indicators
    bb = calculate_bollinger_bands(df)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"] = bb["bb_lower"]
    
    df["rsi"] = RSIIndicator(df["close"]).rsi()
    
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    
    # Convert to list of dicts for JSON serialization
    chart_data = []
    for idx, row in df.iterrows():
        chart_data.append({
            "timestamp": str(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "bb_upper": float(row["bb_upper"]) if pd.notna(row["bb_upper"]) else None,
            "bb_middle": float(row["bb_middle"]) if pd.notna(row["bb_middle"]) else None,
            "bb_lower": float(row["bb_lower"]) if pd.notna(row["bb_lower"]) else None,
            "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
            "macd": float(row["macd"]) if pd.notna(row["macd"]) else None,
            "macd_signal": float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else None,
        })
    
    return chart_data

def calculate_atr(df, period=14):
    """
    Calculate Average True Range for volatility-adjusted stops.
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period (default 14)
    
    Returns:
        Latest ATR value
    """
    try:
        atr = AverageTrueRange(df["high"], df["low"], df["close"], window=period).average_true_range()
        return float(atr.iloc[-1]) if len(atr) > 0 and pd.notna(atr.iloc[-1]) else None
    except:
        return None
