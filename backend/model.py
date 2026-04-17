import xgboost as xgb
import json
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Features in order - must match training order
FEATURES = [
    "bb_position", "bb_width", "rsi", "macd_diff", "volume_ratio",
    "momentum", "volume_change", "ma_signal", "atr_normalized", "roc"
]

def _create_dummy_model():
    """
    Create a dummy trained model for development/testing.
    Uses 10 technical indicator features.
    """
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Create dummy training data with 10 features
    X_dummy = np.random.rand(200, 10)
    X_dummy[:, 0] = np.random.uniform(0, 1, 200)      # BB Position (0-1)
    X_dummy[:, 1] = np.random.uniform(0, 10, 200)     # BB Width (0-10%)
    X_dummy[:, 2] = np.random.uniform(20, 80, 200)    # RSI (20-80)
    X_dummy[:, 3] = np.random.uniform(-5, 5, 200)     # MACD Diff
    X_dummy[:, 4] = np.random.uniform(0.5, 2.0, 200)  # Volume Ratio
    X_dummy[:, 5] = np.random.uniform(-0.05, 0.05, 200)  # Momentum
    X_dummy[:, 6] = np.random.uniform(-0.5, 0.5, 200)    # Volume Change
    X_dummy[:, 7] = np.random.uniform(0, 1, 200)      # MA Signal
    X_dummy[:, 8] = np.random.uniform(0, 0.1, 200)    # ATR Normalized
    X_dummy[:, 9] = np.random.uniform(-0.05, 0.05, 200)  # ROC
    
    y_dummy = np.random.randint(0, 2, 200)
    
    model.fit(X_dummy, y_dummy)
    return model

def load_model():
    """
    Load trained XGBoost model from JSON file.
    Uses Bollinger Bands-based features.
    
    Returns:
        XGBClassifier model object
    """
    try:
        model_path = "trained_model.json"
        
        if os.path.exists(model_path):
            # Load the booster and wrap it in XGBClassifier
            booster = xgb.Booster()
            booster.load_model(model_path)
            
            # Create a new model and set its booster
            model = xgb.XGBClassifier(n_estimators=1)  # Dummy n_estimators
            model._Booster = booster
            model.n_classes_ = 2  # Binary classification
            
            print(f"✓ Loaded trained model from {model_path}")
            return model
        else:
            print(f"⚠ Model file {model_path} not found. Using dummy model for development.")
            print(f"  To train a model, run: python train_model.py")
            return _create_dummy_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Using dummy model for development")
        return _create_dummy_model()

def predict_signal(model, features):
    """
    Predict buy signal probability using 10 technical indicators.
    
    Args:
        model: XGBClassifier model
        features: Dictionary with:
            - bb_position: Position within Bollinger Bands (0-1)
            - bb_width: Width of Bollinger Bands (%)
            - rsi: Relative Strength Index
            - macd_diff: MACD histogram
            - volume_ratio: Volume relative to SMA
            - momentum: 20-period momentum
            - volume_change: Volume change
            - ma_signal: Moving average crossover signal
            - atr_normalized: Normalized ATR volatility
            - roc: 12-period rate of change
        
    Returns:
        Probability of buy signal (0-1)
    """
    try:
        x = np.array([[features[f] for f in FEATURES]])
        prob = model.predict_proba(x)[0][1]
        return prob
    except Exception as e:
        print(f"Error predicting signal: {e}")
        return 0.0
