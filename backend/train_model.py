"""
Train a simpler, short-horizon XGBoost model for more robust live behavior.

- Horizon: 1 bar ahead
- Label: forward return >= 0.5% (with ambiguous +/-0.3% zone dropped)
- Universe: liquid core tickers
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from data_feed import get_historical_bars_for_training
from features import MODEL_FEATURE_COLUMNS, build_feature_matrix

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_model.json"
META_PATH = BASE_DIR / "trained_model_meta.json"

TRAIN_TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA"]
FORWARD_BARS = 1
RETURN_THRESHOLD = 0.005
AMBIGUOUS_BAND = 0.003
ENTRY_THRESHOLD = 0.55


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


def train_model(run_tickers=None):
    print("Building dataset...")
    tickers = (
        [t for t in TRAIN_TICKERS if t in {x.strip().upper() for x in run_tickers}]
        if run_tickers
        else list(TRAIN_TICKERS)
    )

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_ticker: list[str] = []

    spy_bars = None
    try:
        spy_bars_raw = get_historical_bars_for_training("SPY", days=None)
        spy_bars = _normalize_bars_df(spy_bars_raw)
    except Exception:
        spy_bars = None

    for ticker in tickers:
        print(f"  Processing {ticker}...")
        bars_raw = get_historical_bars_for_training(ticker, days=None)
        bars = _normalize_bars_df(bars_raw)
        if bars.empty or len(bars) < 120:
            print("    [WARN] Insufficient data, skipping")
            continue

        use_spy = bars if ticker == "SPY" else spy_bars
        fm = build_feature_matrix(bars, spy_bars=use_spy)
        X = fm.to_numpy(dtype=float)

        forward_ret = (bars["close"].shift(-FORWARD_BARS) - bars["close"]) / bars["close"]
        y = (forward_ret >= RETURN_THRESHOLD).astype(int).to_numpy()
        mask = (np.abs(forward_ret.to_numpy()) > AMBIGUOUS_BAND) & np.isfinite(forward_ret.to_numpy())
        X = X[mask]
        y = y[mask]

        if len(y) == 0:
            print("    [WARN] No usable samples after ambiguity filter")
            continue

        all_X.append(X)
        all_y.append(y)
        all_ticker.extend([ticker] * len(y))
        print(f"    [OK] {len(X)} samples, {(y.sum()/len(y)*100):.1f}% positives")

    if not all_X:
        raise RuntimeError("No training samples built for v2 model.")

    X_base = np.vstack(all_X)
    y = np.hstack(all_y)
    feat_count = len(MODEL_FEATURE_COLUMNS)

    # Add normalized ticker_idx as the final feature to match inference contract.
    ticker_map = {t: i / max(len(TRAIN_TICKERS) - 1, 1) for i, t in enumerate(TRAIN_TICKERS)}
    ticker_idx = np.array([ticker_map.get(t, 0.0) for t in all_ticker], dtype=float).reshape(-1, 1)
    X = np.hstack([X_base, ticker_idx])

    print(f"\nTotal samples: {len(X)}")
    print(f"Feature count: {feat_count + 1} (base={feat_count}, +ticker_idx)")
    print(f"Positives: {(y.sum()/len(y)*100):.1f}% (pos={int(y.sum())}, neg={int((1-y).sum())})")

    n = len(X)
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTraining: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Single-class split encountered in v2 train/test.")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "scale_pos_weight": float((1 - y_train).sum() / max(y_train.sum(), 1)),
        "tree_method": "hist",
        "seed": 42,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("\nTraining XGBoost...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtest, "test")],
        early_stopping_rounds=20,
        verbose_eval=10,
    )

    y_prob = booster.predict(dtest)
    y_pred = (y_prob >= ENTRY_THRESHOLD).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== Test Results ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Score range: {float(y_prob.min()):.4f} to {float(y_prob.max()):.4f}")

    booster.save_model(str(MODEL_PATH))
    print(f"\n[OK] Saved model to {MODEL_PATH}")

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_variant": "v2_promoted",
        "features": list(MODEL_FEATURE_COLUMNS) + ["ticker_idx"],
        "feature_count": int(X.shape[1]),
        "decision_threshold": ENTRY_THRESHOLD,
        "label_mode": "fixed",
        "forward_bars": FORWARD_BARS,
        "return_threshold": RETURN_THRESHOLD,
        "ambiguous_band": AMBIGUOUS_BAND,
        "train_tickers": tickers,
        "metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auc": float(auc),
            "buy_rate": float(y_pred.mean()),
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Saved metadata to {META_PATH}")

    return booster


if __name__ == "__main__":
    train_model()
