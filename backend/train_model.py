"""
Train XGBoost model using only real historical market data.
Fails fast if live historical data is unavailable.
"""

import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score

from features import build_features

warnings.filterwarnings("ignore")

FEATURES = [
    "bb_position",
    "bb_width",
    "rsi",
    "macd_diff",
    "volume_ratio",
    "momentum",
    "volume_change",
    "ma_signal",
    "atr_normalized",
    "roc",
]

TRAIN_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD"]
FORWARD_BARS = 5
RETURN_THRESHOLD = 0.010  # +1.0% forward move threshold
NEGATIVE_RETURN_THRESHOLD = -0.008  # -0.8% forward move threshold
MIN_POSITIVE_PRECISION = 0.52
MAX_BUY_RATE = 0.20
MIN_BUY_RATE = 0.05
TRADE_COST_PCT = 0.08

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.json"
MODEL_META_PATH = BASE_DIR / "trained_model_meta.json"


def _normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Alpaca DataFrame (can be MultiIndex/symbol-indexed) into OHLCV columns.
    """
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

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in bars.columns:
            return pd.DataFrame()

    bars = bars.sort_index()
    bars = bars[required].astype(float)
    return bars


def _build_dataset_for_ticker(ticker: str, days=None):
    try:
        from data_feed import get_historical_bars_for_training
    except Exception:
        return [], []

    bars_raw = get_historical_bars_for_training(ticker, days=days)
    bars = _normalize_bars_df(bars_raw)
    if bars.empty or len(bars) < 80:
        return [], []

    X_rows = []
    y_rows = []
    returns_rows = []

    # Build point-in-time feature vectors and forward-return labels.
    for i in range(60, len(bars) - FORWARD_BARS):
        window = bars.iloc[: i + 1]
        try:
            feat = build_features(window)
        except Exception:
            continue

        current_close = float(bars.iloc[i]["close"])
        future_close = float(bars.iloc[i + FORWARD_BARS]["close"])
        forward_ret = (future_close - current_close) / max(current_close, 1e-9)
        if forward_ret >= RETURN_THRESHOLD:
            label = 1
        elif forward_ret <= NEGATIVE_RETURN_THRESHOLD:
            label = 0
        else:
            # Drop weak/ambiguous moves to improve separability.
            continue

        X_rows.append([feat.get(name, 0.0) for name in FEATURES])
        y_rows.append(label)
        returns_rows.append(float(forward_ret))

    return X_rows, y_rows, returns_rows


def _time_split_3way(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
        r[:train_end],
        r[train_end:val_end],
        r[val_end:],
    )


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, forward_returns: np.ndarray):
    best_t = 0.70
    best_score = -1e9
    for t in np.linspace(0.30, 0.90, 61):
        y_pred = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        buy_rate = float(y_pred.mean())
        realized = forward_returns[y_pred == 1] if y_pred.any() else np.array([])
        avg_trade_return = float(realized.mean()) if realized.size else -0.01
        expectancy = avg_trade_return - (TRADE_COST_PCT / 100.0)

        # Strongly favor high precision and sparse high-confidence entries.
        score = (
            1.35 * precision
            + 0.55 * bal_acc
            + 0.35 * f1
            + 1.50 * expectancy
            - 0.55 * max(0.0, buy_rate - MAX_BUY_RATE)
            - 0.50 * max(0.0, MIN_BUY_RATE - buy_rate)
            - 0.40 * max(0.0, MIN_POSITIVE_PRECISION - precision)
        )
        if recall < 0.10:
            score -= 0.20

        if score > best_score:
            best_score = score
            best_t = float(t)
    if best_t >= 0.88:
        return 0.68
    return best_t


def _walk_forward_threshold(X_train: np.ndarray, y_train: np.ndarray, r_train: np.ndarray, folds: int = 4):
    n = len(X_train)
    fold_size = n // (folds + 1)
    candidates = []
    for i in range(1, folds + 1):
        train_end = fold_size * i
        val_end = fold_size * (i + 1)
        if val_end > n or train_end < 300:
            break

        X_tr, y_tr = X_train[:train_end], y_train[:train_end]
        X_val, y_val = X_train[train_end:val_end], y_train[train_end:val_end]
        r_val = r_train[train_end:val_end]
        if len(X_val) < 100 or len(np.unique(y_tr)) < 2:
            continue

        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        spw = max(1.0, neg / max(pos, 1))
        fold_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.70,
            colsample_bytree=0.65,
            reg_lambda=4.0,
            reg_alpha=1.2,
            min_child_weight=8,
            gamma=1.0,
            random_state=42 + i,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        fold_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_prob_val = fold_model.predict_proba(X_val)[:, 1]
        candidates.append(_find_best_threshold(y_val, y_prob_val, r_val))

    if not candidates:
        return 0.70
    return float(np.median(candidates))


def train_model():
    print("Collecting training data...")

    X_all = []
    y_all = []
    r_all = []
    per_ticker_counts = {}
    failed_tickers = []

    for ticker in TRAIN_TICKERS:
        X_t, y_t, r_t = _build_dataset_for_ticker(ticker, days=None)
        if X_t and y_t and r_t:
            X_all.extend(X_t)
            y_all.extend(y_t)
            r_all.extend(r_t)
            per_ticker_counts[ticker] = len(y_t)
        else:
            failed_tickers.append(ticker)

    if not X_all:
        raise RuntimeError(
            "No real historical training data loaded from Alpaca. "
            "Synthetic fallback is disabled. Check credentials/data access."
        )

    X = np.array(X_all, dtype=float)
    y = np.array(y_all, dtype=int)
    r = np.array(r_all, dtype=float)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        r_train,
        r_val,
        r_test,
    ) = _time_split_3way(X, y, r, train_ratio=0.7, val_ratio=0.15)

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training labels are single-class; cannot train classifier.")
    if len(X_val) < 100 or len(X_test) < 100:
        raise RuntimeError("Insufficient validation/test samples for robust evaluation.")

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, neg / max(pos, 1))

    walk_forward_threshold = _walk_forward_threshold(X_train, y_train, r_train, folds=4)

    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.02,
        subsample=0.70,
        colsample_bytree=0.65,
        reg_lambda=4.0,
        reg_alpha=1.2,
        min_child_weight=8,
        gamma=1.0,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Tune threshold ONLY on validation set (no test leakage).
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_threshold = _find_best_threshold(y_val, y_prob_val, r_val)
    best_threshold = float(np.clip((walk_forward_threshold + val_threshold) / 2.0, 0.40, 0.92))

    # Final unbiased test evaluation.
    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob_test >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0.5
    buy_rate = float(y_pred.mean())
    selected_returns = r_test[y_pred == 1] if y_pred.any() else np.array([])
    avg_trade_return = float(selected_returns.mean()) if selected_returns.size else 0.0

    print("Training complete.")
    print(f"Samples: total={len(X)} train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"Class balance (train): pos={pos} neg={neg} scale_pos_weight={scale_pos_weight:.2f}")
    print(f"Metrics: acc={acc:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(
        f"Thresholds: walk_forward={walk_forward_threshold:.3f} "
        f"validation={val_threshold:.3f} final={best_threshold:.3f}"
    )
    print(f"Predicted BUY rate on holdout: {buy_rate:.3f}")
    print(f"Avg selected forward return (gross): {avg_trade_return*100:.3f}%")
    print(f"Per-ticker sample counts: {per_ticker_counts}")
    if failed_tickers:
        print(f"Tickers with insufficient/unavailable history: {failed_tickers}")

    model.get_booster().save_model(str(MODEL_PATH))

    metadata = {
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "decision_threshold": best_threshold,
        "forward_bars": FORWARD_BARS,
        "return_threshold": RETURN_THRESHOLD,
        "negative_return_threshold": NEGATIVE_RETURN_THRESHOLD,
        "min_positive_precision_target": MIN_POSITIVE_PRECISION,
        "max_buy_rate_target": MAX_BUY_RATE,
        "min_buy_rate_target": MIN_BUY_RATE,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "buy_rate": float(buy_rate),
            "avg_selected_forward_return": float(avg_trade_return),
        },
        "split": {
            "train_size": int(len(X_train)),
            "validation_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "train_tickers": TRAIN_TICKERS,
        "per_ticker_counts": per_ticker_counts,
        "failed_tickers": failed_tickers,
    }

    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {MODEL_META_PATH}")
    return model


if __name__ == "__main__":
    train_model()
