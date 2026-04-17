"""
Train XGBoost model using Bollinger Bands and technical indicators.
This script generates synthetic training data and saves a trained model.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_training_data(n_samples=5000):
    """
    Generate synthetic training data based on realistic market patterns.
    Features include: Bollinger Bands, RSI, MACD, Volume, Momentum, ATR, ROC
    """
    np.random.seed(42)
    
    # Generate synthetic price data with trends and mean reversion
    returns = np.random.normal(0.0005, 0.02, n_samples)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # === Calculate all indicators ===
    
    # Bollinger Bands (20-period, 2 std)
    window = 20
    sma = pd.Series(prices).rolling(window=window).mean()
    std = pd.Series(prices).rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    bb_position = np.nan_to_num(bb_position, nan=0.5)
    bb_position = np.clip(bb_position, 0, 1)
    bb_width = (upper_band - lower_band) / prices
    bb_width = np.nan_to_num(bb_width, nan=0.1)
    
    # RSI (14-period)
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50)
    
    # MACD
    ema12 = pd.Series(prices).ewm(span=12).mean()
    ema26 = pd.Series(prices).ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    macd_diff = macd - signal
    
    # Volume (synthetic)
    volume = np.random.uniform(1000000, 5000000, n_samples)
    volume_sma = pd.Series(volume).rolling(window=20).mean()
    volume_ratio = volume / (volume_sma + 1e-10)
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    
    # === NEW FEATURES ===
    
    # 1. Momentum (20-period price change %)
    momentum = np.zeros(n_samples)
    for i in range(20, n_samples):
        momentum[i] = (prices[i] - prices[i-20]) / prices[i-20]
    
    # 2. Volume Change (5-period)
    volume_change = np.zeros(n_samples)
    for i in range(5, n_samples):
        vol_prev = volume[i-5] if volume[i-5] > 0 else 1e-10
        volume_change[i] = (volume[i] - vol_prev) / vol_prev
    
    # 3. Moving Average Signal (SMA10 > SMA30)
    sma10 = pd.Series(prices).rolling(window=10).mean()
    sma30 = pd.Series(prices).rolling(window=30).mean()
    ma_signal = (sma10 > sma30).astype(float)
    ma_signal = np.nan_to_num(ma_signal, nan=0)
    
    # 4. ATR Normalized (volatility)
    atr_normalized = std / prices
    atr_normalized = np.nan_to_num(atr_normalized, nan=0.01)
    
    # 5. ROC (12-period Rate of Change)
    roc = np.zeros(n_samples)
    for i in range(12, n_samples):
        roc[i] = (prices[i] - prices[i-12]) / prices[i-12]
    
    # Combine all 10 features
    X = np.column_stack([
        bb_position,      # Position within Bollinger Bands
        bb_width,         # Width of Bollinger Bands
        rsi,              # RSI indicator
        macd_diff,        # MACD histogram
        volume_ratio,     # Volume ratio
        momentum,         # 20-period momentum
        volume_change,    # Volume change
        ma_signal,        # MA crossover signal
        atr_normalized,   # Normalized ATR volatility
        roc               # 12-period rate of change
    ])
    
    # IMPROVED LABELS: 1 if future return > 2%, 0 otherwise (more realistic trading threshold)
    future_prices = np.roll(prices, -5)
    future_return = (future_prices - prices) / prices
    labels = (future_return > 0.02).astype(int)
    labels[-5:] = labels[-6]  # Fix the last few values
    
    # Remove NaN rows
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X[valid_idx]
    labels = labels[valid_idx]
    
    return X, labels

def train_model():
    """Train XGBoost model with improved features and labels."""
    print("📊 Generating training data with enhanced features...")
    X, y = generate_training_data(n_samples=5000)
    
    print(f"✓ Generated {len(X)} training samples")
    print(f"  Features: 10 technical indicators")
    print(f"  - Original: BB Position, BB Width, RSI, MACD, Volume Ratio")
    print(f"  - Enhanced: Momentum, Volume Change, MA Signal, ATR, ROC")
    buy_count = np.sum(y)
    print(f"  Labels: {buy_count} strong signals (>2% return), {len(y) - buy_count} weak signals")
    
    # Split data with stratification for better validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with optimized hyperparameters
    print("\n🤖 Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,           # Increased to handle 10 features
        max_depth=4,                # Slightly deeper for more patterns
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        reg_alpha=0.5,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"✓ Model trained!")
    print(f"  Train accuracy: {train_score:.2%}")
    print(f"  Test accuracy: {test_score:.2%}")
    print(f"  Overfitting gap: {(train_score - test_score):.2%}")
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"  5-fold CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    # Feature importance
    importance = model.feature_importances_
    features = [
        'BB Position', 'BB Width', 'RSI', 'MACD', 'Volume Ratio',
        'Momentum', 'Volume Change', 'MA Signal', 'ATR', 'ROC'
    ]
    print(f"\n📈 Feature importance:")
    for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.2%}")
    
    # Save model
    model.get_booster().save_model('trained_model.json')
    print(f"\n✓ Model saved to: trained_model.json")
    
    return model

if __name__ == "__main__":
    train_model()
