"""
Backtest the trading strategy on real historical data only.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path

from data_feed import get_historical_bars_for_training

FORWARD_BARS = 5
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
    
    return df

def backtest_strategy(model, df, threshold=0.5):
    """
    Backtest the trading strategy.
    
    Returns:
        Dictionary with performance metrics
    """
    df = calculate_indicators(df)
    df = df.dropna()
    
    # Generate predictions with 10 features
    features = [
        'bb_position', 'bb_width', 'rsi', 'macd_diff', 'volume_ratio',
        'momentum', 'volume_change', 'ma_signal', 'atr_normalized', 'roc'
    ]
    X = df[features].values
    
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
    results = backtest_strategy(model, df, threshold=threshold)
    
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
