from typing import Optional

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange

def serialize_timestamp(index_value):
    """
    Normalize DataFrame index values to ISO timestamp strings.
    Handles MultiIndex rows returned by some Alpaca responses.
    """
    ts_value = index_value[-1] if isinstance(index_value, tuple) else index_value
    try:
        return pd.Timestamp(ts_value).isoformat()
    except Exception:
        return str(ts_value)

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

# Order must match training / model inference (base columns; ticker_idx appended in train_model).
MODEL_FEATURE_COLUMNS = (
    "bb_position",
    "bb_width",
    "rsi",
    "rsi_distance_50",
    "macd_diff",
    "macd_slope_3",
    "macd_cross_up",
    "ema_sma_spread",
    "price_vs_ema20",
    "price_vs_sma20",
    "ma_signal",
    "trend_strength",
    "ret_1",
    "ret_5",
    "ret_20",
    "hl_range_pct",
    "close_in_range",
    "atr_normalized",
    "gap_1",
    "ret_vol_20",
    "vol_ratio_20_60",
    "volume_ratio",
    "volume_change",
    "volume_z_20",
    "signed_volume_ratio_20",
    "obv_slope_5",
    "vwap_dev",
    "vwap_z_20",
    "vol_profile_dev",
    "vol_profile_z_20",
    "mean_rev_z_20",
    "mean_rev_bb",
    "mkt_ret_1",
    "mkt_ret_5",
    "mkt_ret_10",
    "rel_strength_5",
    "rel_strength_10",
    "corr_spy_20",
)

def _merge_spy_context(df: pd.DataFrame, spy_bars: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Align SPY OHLCV to df.index for market-relative and cross-asset features."""
    if spy_bars is None or len(spy_bars) == 0 or "close" not in spy_bars.columns:
        df["mkt_ret_1"] = 0.0
        df["mkt_ret_5"] = 0.0
        df["mkt_ret_10"] = 0.0
        df["rel_strength_5"] = 0.0
        df["rel_strength_10"] = 0.0
        df["corr_spy_20"] = 0.0
        return df
    spy = spy_bars.copy()
    if not spy.index.equals(df.index):
        spy_close = spy["close"].reindex(df.index).ffill()
    else:
        spy_close = spy["close"]
    m1 = spy_close.pct_change(1)
    m5 = spy_close.pct_change(5)
    m10 = spy_close.pct_change(10)
    r_stock_1 = df["close"].pct_change(1)
    df["mkt_ret_1"] = m1.fillna(0.0).astype(float)
    df["mkt_ret_5"] = m5.fillna(0.0).astype(float)
    df["mkt_ret_10"] = m10.fillna(0.0).astype(float)
    df["rel_strength_5"] = (df["ret_5"] - m5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["rel_strength_10"] = (df["close"].pct_change(10) - m10).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["corr_spy_20"] = r_stock_1.rolling(20, min_periods=10).corr(m1)
    df["corr_spy_20"] = df["corr_spy_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _compute_indicator_dataframe(
    df: pd.DataFrame, spy_bars: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Full causal indicator frame (same values as point-in-time `build_features` at each row)."""
    df = df.copy()
    if "open" not in df.columns:
        df["open"] = df["close"]

    bb = calculate_bollinger_bands(df)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"] = bb["bb_lower"]
    df["bb_position"] = bb["bb_position"]
    df["bb_width"] = bb["bb_width"]

    df["rsi"] = RSIIndicator(df["close"]).rsi()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    atr_series = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr"] = atr_series
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["macd_diff"] = df["macd"] - df["macd_signal"]

    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)

    df["momentum"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20)
    df["momentum"] = df["momentum"].fillna(0)

    df["volume_change"] = (df["volume"] - df["volume"].shift(5)) / (df["volume"].shift(5) + 1e-10)
    df["volume_change"] = df["volume_change"].fillna(0)

    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_30"] = df["close"].rolling(window=30).mean()
    df["ma_signal"] = (df["sma_10"] > df["sma_30"]).astype(float)
    df["ma_signal"] = df["ma_signal"].fillna(0)

    df["atr_normalized"] = df["atr"] / df["close"]
    df["atr_normalized"] = df["atr_normalized"].fillna(0)

    df["rsi_distance_50"] = (df["rsi"] - 50.0) / 50.0
    df["rsi_slope_3"] = (df["rsi"] - df["rsi"].shift(3)) / 3.0
    df["ema_sma_spread"] = (df["ema_20"] - df["sma_20"]) / (df["close"] + 1e-10)
    df["price_vs_ema20"] = (df["close"] - df["ema_20"]) / (df["close"] + 1e-10)
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / (df["close"] + 1e-10)
    df["macd_slope_3"] = (df["macd_diff"] - df["macd_diff"].shift(3)) / 3.0
    df["macd_cross_up"] = (df["macd_diff"] > 0).astype(float)
    df["trend_strength"] = np.abs(df["ema_20"] - df["sma_20"]) / (df["atr"] + 1e-10)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["hl_range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_in_range"] = (df["close"] - df["low"]) / (rng + 1e-10)
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["volume_z_20"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)

    df["gap_1"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)
    df["gap_1"] = df["gap_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    r1 = df["close"].pct_change(1)
    df["ret_vol_20"] = r1.rolling(window=20, min_periods=10).std()
    df["ret_vol_20"] = df["ret_vol_20"].fillna(0.0)
    sma50 = df["close"].rolling(window=50, min_periods=25).mean()
    df["dist_sma50"] = (df["close"] - sma50) / (df["close"] + 1e-10)
    df["dist_sma50"] = df["dist_sma50"].fillna(0.0)
    # Approximate order-flow pressure from candle direction and volume.
    signed_volume = np.sign(df["close"] - df["open"]) * df["volume"]
    df["signed_volume_ratio_20"] = signed_volume.rolling(20, min_periods=10).sum() / (
        df["volume"].rolling(20, min_periods=10).sum() + 1e-10
    )
    df["signed_volume_ratio_20"] = df["signed_volume_ratio_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # OBV slope is another lightweight flow feature.
    obv = (np.sign(df["close"].diff()).fillna(0.0) * df["volume"]).cumsum()
    df["obv_slope_5"] = (obv - obv.shift(5)) / (df["volume"].rolling(20, min_periods=10).mean() * 5 + 1e-10)
    df["obv_slope_5"] = df["obv_slope_5"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # VWAP / market microstructure deviations.
    df["vwap_dev"] = (df["close"] - df["vwap"]) / (df["close"] + 1e-10)
    vwap_std = df["vwap_dev"].rolling(20, min_periods=10).std()
    df["vwap_z_20"] = (df["vwap_dev"] - df["vwap_dev"].rolling(20, min_periods=10).mean()) / (vwap_std + 1e-10)
    df["vwap_z_20"] = df["vwap_z_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Volume profile proxy: rolling volume-weighted typical price.
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vp_window = 30
    vol_sum = df["volume"].rolling(vp_window, min_periods=10).sum()
    vol_typ_sum = (typical_price * df["volume"]).rolling(vp_window, min_periods=10).sum()
    vol_profile_price = vol_typ_sum / (vol_sum + 1e-10)
    df["vol_profile_dev"] = (df["close"] - vol_profile_price) / (df["close"] + 1e-10)
    vp_std = df["vol_profile_dev"].rolling(20, min_periods=10).std()
    df["vol_profile_z_20"] = df["vol_profile_dev"] / (vp_std + 1e-10)
    df["vol_profile_z_20"] = df["vol_profile_z_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Mean-reversion metrics (overbought / oversold).
    px_std_20 = df["close"].rolling(20, min_periods=10).std()
    df["mean_rev_z_20"] = (df["close"] - df["sma_20"]) / (px_std_20 + 1e-10)
    df["mean_rev_z_20"] = df["mean_rev_z_20"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["mean_rev_bb"] = (df["bb_position"] - 0.5) * 2.0
    df["mean_rev_bb"] = df["mean_rev_bb"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    v20 = r1.rolling(20, min_periods=10).std()
    v60 = r1.rolling(60, min_periods=30).std()
    df["vol_ratio_20_60"] = v20 / (v60 + 1e-10)
    df["vol_ratio_20_60"] = df["vol_ratio_20_60"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df = _merge_spy_context(df, spy_bars)

    return df


def build_feature_matrix(
    bars: pd.DataFrame, spy_bars: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Vectorized features for training: one row per bar, strictly causal."""
    d = _compute_indicator_dataframe(bars, spy_bars=spy_bars)
    out = pd.DataFrame({c: d[c].values for c in MODEL_FEATURE_COLUMNS}, index=bars.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(df, spy_bars: Optional[pd.DataFrame] = None):
    """
    Build enhanced technical indicators from OHLCV data (latest bar only).

    Returns:
        Dictionary with computed features for ML model
    """
    d = _compute_indicator_dataframe(df, spy_bars=spy_bars)
    nonempty = d.dropna(how="all")
    latest = nonempty.iloc[-1] if len(nonempty) > 0 else d.iloc[-1]

    def _f(name: str, default: float = 0.0) -> float:
        v = latest.get(name, default)
        if pd.isna(v):
            return default
        return float(v)

    out = {name: _f(name, 0.5 if name == "close_in_range" else 0.0) for name in MODEL_FEATURE_COLUMNS}
    return out

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
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # Convert to list of dicts for JSON serialization
    chart_data = []
    for idx, row in df.iterrows():
        chart_data.append({
            "timestamp": serialize_timestamp(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "bb_upper": float(row["bb_upper"]) if pd.notna(row["bb_upper"]) else None,
            "bb_middle": float(row["bb_middle"]) if pd.notna(row["bb_middle"]) else None,
            "bb_lower": float(row["bb_lower"]) if pd.notna(row["bb_lower"]) else None,
            "ema_20": float(row["ema_20"]) if "ema_20" in row and pd.notna(row["ema_20"]) else None,
            "sma_20": float(row["sma_20"]) if "sma_20" in row and pd.notna(row["sma_20"]) else None,
            "vwap": float(row["vwap"]) if "vwap" in row and pd.notna(row["vwap"]) else None,
            "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else None,
            "macd": float(row["macd"]) if pd.notna(row["macd"]) else None,
            "macd_signal": float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else None,
            "atr": float(row["atr"]) if "atr" in row and pd.notna(row["atr"]) else None,
            "stoch_k": float(row["stoch_k"]) if "stoch_k" in row and pd.notna(row["stoch_k"]) else None,
            "stoch_d": float(row["stoch_d"]) if "stoch_d" in row and pd.notna(row["stoch_d"]) else None,
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
