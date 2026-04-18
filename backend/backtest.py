"""
Backtest the trading strategy on real historical data only.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from ta.volatility import AverageTrueRange

from data_feed import get_historical_bars_for_training
from train_model import TRAIN_TICKERS
from model import FEATURES
from features import HORIZON_BARS

# Match train_model.FORWARD_BARS for comparable forward-return horizon in backtests.
FORWARD_BARS = 10
TRADE_COST_PCT = 0.08  # Approx round-trip cost in percent (slippage + fees)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.json"
MODEL_META_PATH = BASE_DIR / "trained_model_meta.json"

def _normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    bars = df.copy()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.reset_index()
        if "timestamp" in bars.columns:
            bars = bars.set_index("timestamp")
        for candidate in ["symbol", "ticker"]:
            if candidate in bars.columns:
                bars = bars.drop(columns=[candidate])
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in bars.columns:
            return pd.DataFrame()
    return bars[required].astype(float).sort_index()

def calculate_indicators(df):
    """Calculate all technical indicators."""
    df = df.copy()
    
    # Bollinger Bands
    window = 20
    sma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_upper"] = sma + (std * 2)
    df["bb_lower"] = sma - (std * 2)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_position"] = df["bb_position"].clip(0, 1)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"] * 100
    
    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_diff"] = macd - signal
    
    # Volume Ratio
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma"] + 1e-10)
    
    # NEW FEATURES
    # Momentum (20-period)
    df["momentum"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20)
    df["momentum"] = df["momentum"].fillna(0)
    
    # Volume Change (5-period)
    df["volume_change"] = (df["volume"] - df["volume"].shift(5)) / (df["volume"].shift(5) + 1e-10)
    df["volume_change"] = df["volume_change"].fillna(0)
    
    # MA Signal (SMA10 > SMA30)
    df["sma10"] = df["close"].rolling(window=10).mean()
    df["sma30"] = df["close"].rolling(window=30).mean()
    df["ma_signal"] = (df["sma10"] > df["sma30"]).astype(float)
    df["ma_signal"] = df["ma_signal"].fillna(0)
    
    # ATR Normalized
    df["atr_normalized"] = std / df["close"]
    df["atr_normalized"] = df["atr_normalized"].fillna(0.01)
    
    # ROC (12-period)
    df["roc"] = (df["close"] - df["close"].shift(12)) / df["close"].shift(12)
    df["roc"] = df["roc"].fillna(0)

    ema20 = df["close"].ewm(span=20).mean()
    sma20 = df["close"].rolling(window=20).mean()

    # RSI / MA / MACD derived factors
    df["rsi_distance_50"] = (df["rsi"] - 50.0) / 50.0
    df["rsi_slope_3"] = (df["rsi"] - df["rsi"].shift(3)) / 3.0
    df["ema_sma_spread"] = (ema20 - sma20) / (df["close"] + 1e-10)
    df["price_vs_ema20"] = (df["close"] - ema20) / (df["close"] + 1e-10)
    df["price_vs_sma20"] = (df["close"] - sma20) / (df["close"] + 1e-10)
    df["macd_slope_3"] = (df["macd_diff"] - df["macd_diff"].shift(3)) / 3.0
    df["macd_cross_up"] = (df["macd_diff"] > 0).astype(float)
    atr_series = AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["trend_strength"] = np.abs(ema20 - sma20) / (atr_series + 1e-10)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["hl_range_pct"] = (df["high"] - df["low"]) / (df["close"] + 1e-10)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_in_range"] = (df["close"] - df["low"]) / (rng + 1e-10)
    vol_mean = df["volume"].rolling(window=20).mean()
    vol_std = df["volume"].rolling(window=20).std()
    df["volume_z_20"] = (df["volume"] - vol_mean) / (vol_std + 1e-10)

    df["mom_risk"] = df["momentum"] / (df["atr_normalized"] + 0.01)
    df["gap_1"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)
    df["gap_1"] = df["gap_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    r1 = df["close"].pct_change(1)
    df["ret_vol_20"] = r1.rolling(window=20, min_periods=10).std().fillna(0.0)
    sma50_bt = df["close"].rolling(window=50, min_periods=25).mean()
    df["dist_sma50"] = ((df["close"] - sma50_bt) / (df["close"] + 1e-10)).fillna(0.0)
    roll_hi = df["close"].rolling(window=126, min_periods=50).max()
    df["dd_126"] = ((df["close"] - roll_hi) / (roll_hi + 1e-10)).fillna(0.0)
    df["up_days_5"] = r1.gt(0).astype(float).rolling(window=5, min_periods=1).sum().fillna(0.0)

    sig20 = r1.rolling(window=20, min_periods=10).std()
    ret_h = df["close"].pct_change(HORIZON_BARS)
    df["mom_10_vol"] = (ret_h / (sig20 * np.sqrt(float(HORIZON_BARS)) + 1e-8)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    df["align_20_50"] = (
        np.sign(df["close"] - sma20) * np.sign(df["close"] - sma50_bt)
    ).fillna(0.0)
    rng_bar = df["hl_range_pct"]
    df["range_z_20"] = (rng_bar / (rng_bar.rolling(20, min_periods=10).mean() + 1e-8)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(1.0)
    roll_hi_63 = df["close"].rolling(window=63, min_periods=30).max()
    df["dd_pct_63"] = ((df["close"] - roll_hi_63) / (roll_hi_63 + 1e-10)).fillna(0.0)
    v20 = r1.rolling(20, min_periods=10).std()
    v60 = r1.rolling(60, min_periods=30).std()
    df["vol_ratio_20_60"] = (v20 / (v60 + 1e-10)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    # run_backtest uses SPY only — market return proxy = asset return
    df["mkt_ret_1"] = df["ret_1"]
    df["mkt_ret_5"] = df["ret_5"]

    return df

def backtest_strategy(model, df, threshold=0.5, ticker="SPY"):
    """
    Backtest the trading strategy.
    
    Returns:
        Dictionary with performance metrics
    """
    df = calculate_indicators(df)
    df = df.dropna()
    tid = (
        TRAIN_TICKERS.index(ticker) / max(len(TRAIN_TICKERS) - 1, 1)
        if ticker in TRAIN_TICKERS
        else 0.0
    )
    df["ticker_idx"] = float(tid)

    X = df[FEATURES].values
    
    predictions = model.predict_proba(X)[:, 1]  # Probability of BUY
    signals = (predictions >= float(threshold)).astype(int)  # 1 = BUY, 0 = WAIT
    
    df['prediction'] = predictions
    df['signal'] = signals
    
    # Calculate returns
    df['price_change'] = df['close'].pct_change() * 100
    df["future_return"] = (df["close"].shift(-FORWARD_BARS) / df["close"] - 1.0) * 100
    
    # Calculate strategy returns
    df["strategy_return_gross"] = df["signal"] * df["future_return"]
    df["strategy_return"] = np.where(
        df["signal"] == 1,
        df["strategy_return_gross"] - TRADE_COST_PCT,
        0.0,
    )
    
    # Metrics
    buy_signals = signals.sum()
    buy_positions = df[df['signal'] == 1]
    
    if len(buy_positions) > 0:
        avg_signal_prob = buy_positions['prediction'].mean()
        avg_buy_return = buy_positions['future_return'].mean()
        win_rate = (buy_positions['future_return'] > 0).sum() / len(buy_positions) * 100
        total_return = df["strategy_return"].sum()
        cumulative_return = (1 + df["strategy_return"] / 100).prod() - 1
    else:
        avg_signal_prob = 0
        avg_buy_return = 0
        win_rate = 0
        total_return = 0
        cumulative_return = 0
    
    # Buy and hold comparison
    buy_hold_return = (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
    
    return {
        'total_candles': len(df),
        'buy_signals': int(buy_signals),
        "buy_rate": round(float(buy_signals / max(len(df), 1)), 4),
        'avg_signal_probability': round(avg_signal_prob, 2),
        'avg_buy_return': round(avg_buy_return, 2),
        'win_rate': round(win_rate, 2),
        'total_strategy_return': round(total_return, 2),
        'cumulative_strategy_return': round(cumulative_return * 100, 2),
        'buy_hold_return': round(buy_hold_return * 100, 2),
        'strategy_outperformance': round(cumulative_return * 100 - buy_hold_return * 100, 2),
    }

def run_backtest():
    """Run complete backtest."""
    ticker = "SPY"
    print(f"Loading full available historical data for {ticker} from Alpaca...")
    df_raw = get_historical_bars_for_training(ticker, days=None)
    df = _normalize_bars_df(df_raw)
    if df.empty:
        raise RuntimeError(
            f"No real historical data returned for {ticker}. "
            "Synthetic data is disabled for backtesting."
        )
    
    print("Loading trained model...")
    try:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        # Create a new model and set its booster
        model = xgb.XGBClassifier(n_estimators=1)  # Dummy n_estimators
        model._Booster = booster
        model.n_classes_ = 2  # Binary classification
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Run 'python train_model.py' first.")
        return
    
    threshold = 0.5
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            threshold = float(meta.get("decision_threshold", 0.5))
    except Exception:
        pass

    print("\nRunning backtest on real historical data...")
    results = backtest_strategy(model, df, threshold=threshold, ticker=ticker)
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total candles analyzed: {results['total_candles']}")
    print(f"Buy signals generated: {results['buy_signals']}")
    print(f"Buy rate: {results['buy_rate']}")
    print(f"Average signal probability: {results['avg_signal_probability']}")
    print(f"\nPerformance Metrics:")
    print(f"  Average buy return: {results['avg_buy_return']}%")
    print(f"  Win rate: {results['win_rate']}%")
    print(f"  Total strategy return: {results['total_strategy_return']}%")
    print(f"  Cumulative strategy return: {results['cumulative_strategy_return']}%")
    print(f"  Buy & hold return: {results['buy_hold_return']}%")
    print(f"  Strategy outperformance: {results['strategy_outperformance']}%")
    print("="*60)
    
    # Interpretation
    print("\nInterpretation:")
    if results['strategy_outperformance'] > 0:
        print(f"Strategy outperforms buy & hold by {results['strategy_outperformance']}%")
    else:
        print(f"Strategy underperforms buy & hold by {abs(results['strategy_outperformance'])}%")
    
    if results['win_rate'] > 50:
        print(f"Win rate ({results['win_rate']}%) is above 50%")
    else:
        print(f"Win rate ({results['win_rate']}%) needs improvement")

if __name__ == "__main__":
    run_backtest()
