import xgboost as xgb
import json
import numpy as np
import os
from pathlib import Path

# Features in order - must match training order
FEATURES = [
    "bb_position", "bb_width", "rsi", "macd_diff", "volume_ratio",
    "momentum", "volume_change", "ma_signal", "atr_normalized", "roc"
]

# Backward-compatible 5-feature order used by older trained models.
LEGACY_FEATURES_5 = [
    "rsi", "macd_diff", "volume_ratio", "momentum", "atr_normalized"
]

DEFAULT_DECISION_THRESHOLD = 0.50
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.json"
MODEL_META_PATH = BASE_DIR / "trained_model_meta.json"

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
        model_path = MODEL_PATH
        
        if model_path.exists():
            # Load the booster and wrap it in XGBClassifier
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            
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

def load_decision_threshold(default_value=DEFAULT_DECISION_THRESHOLD):
    """
    Load inference threshold from training metadata when available.
    """
    try:
        meta_path = MODEL_META_PATH
        if not meta_path.exists():
            return float(default_value)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        threshold = float(meta.get("decision_threshold", default_value))
        return max(0.0, min(1.0, threshold))
    except Exception as e:
        print(f"Warning: failed to load decision threshold: {e}")
        return float(default_value)

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
        expected_feature_count = None
        try:
            expected_feature_count = int(model.get_booster().num_features())
        except Exception:
            expected_feature_count = None

        if expected_feature_count == 5:
            selected_features = LEGACY_FEATURES_5
            x_values = [features.get(f, 0.0) for f in selected_features]
        elif expected_feature_count is None:
            selected_features = FEATURES
            x_values = [features.get(f, 0.0) for f in selected_features]
        else:
            # Build exactly expected_feature_count values to avoid booster index errors.
            # Fill known feature slots first, then pad remaining slots with 0.
            base_values = [features.get(f, 0.0) for f in FEATURES]
            if expected_feature_count <= len(base_values):
                x_values = base_values[:expected_feature_count]
            else:
                x_values = base_values + [0.0] * (expected_feature_count - len(base_values))

        x = np.array([x_values], dtype=float)

        # Prefer native booster prediction for loaded JSON boosters to avoid wrapper quirks.
        try:
            booster = model.get_booster()
            dmatrix = xgb.DMatrix(x)
            booster_pred = booster.predict(dmatrix)
            if len(booster_pred) > 0:
                prob = float(booster_pred[0])
                if prob < 0:
                    prob = 1.0 / (1.0 + np.exp(-prob))
                return max(0.0, min(1.0, prob))
        except Exception:
            pass

        prob = float(model.predict_proba(x)[0][1])
        return max(0.0, min(1.0, prob))
    except Exception as e:
        print(f"Error predicting signal: {e}")
        return 0.0
