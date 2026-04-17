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

from features import build_feature_matrix, MODEL_FEATURE_COLUMNS

warnings.filterwarnings("ignore")

FEATURES_BASE = list(MODEL_FEATURE_COLUMNS)
FEATURES = FEATURES_BASE + ["ticker_idx"]

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
        return []

    bars_raw = get_historical_bars_for_training(ticker, days=days)
    bars = _normalize_bars_df(bars_raw)
    if bars.empty or len(bars) < 80:
        return []

    tid = TRAIN_TICKERS.index(ticker) / max(len(TRAIN_TICKERS) - 1, 1)
    try:
        fm = build_feature_matrix(bars)
    except Exception:
        return []
    mat = fm.to_numpy(dtype=float)
    forward_ret_ser = (bars["close"].shift(-FORWARD_BARS) - bars["close"]) / bars["close"]
    forward_ret = forward_ret_ser.to_numpy()
    rows = []
    n = len(bars)
    for i in range(60, n - FORWARD_BARS):
        fr = forward_ret[i]
        if not np.isfinite(fr):
            continue
        if fr >= RETURN_THRESHOLD:
            label = 1
        elif fr <= NEGATIVE_RETURN_THRESHOLD:
            label = 0
        else:
            continue
        vec = mat[i].tolist()
        vec.append(float(tid))
        ts = int(pd.Timestamp(bars.index[i]).value)
        rows.append((ts, vec, label, float(fr)))

    return rows


def _merge_time_sorted_rows(parts):
    rows = []
    for p in parts:
        rows.extend(p)
    rows.sort(key=lambda x: x[0])
    if not rows:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
        )
    X = np.array([r[1] for r in rows], dtype=float)
    y = np.array([r[2] for r in rows], dtype=int)
    r = np.array([r[3] for r in rows], dtype=float)
    return X, y, r


def _train_xgb_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    scale_pos_weight,
    *,
    random_state,
    feature_names=None,
    num_boost_round=3000,
    early_stopping_rounds=200,
    max_depth=4,
    eta=0.025,
    subsample=0.75,
    colsample_bytree=0.75,
    reg_lambda=2.5,
    reg_alpha=0.8,
    min_child_weight=5,
    gamma=0.15,
):
    """
    XGBoost 3.x sklearn wrapper dropped early_stopping_rounds on fit(); use native train().
    Returns an XGBClassifier with _Booster set for save_model / predict_proba.
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": int(max_depth),
        "eta": float(eta),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "reg_lambda": float(reg_lambda),
        "reg_alpha": float(reg_alpha),
        "min_child_weight": float(min_child_weight),
        "gamma": float(gamma),
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": "hist",
        "seed": int(random_state),
    }
    fn = list(feature_names) if feature_names is not None else None
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=fn)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=fn)
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dval, "validation")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    clf = xgb.XGBClassifier()
    clf._Booster = bst
    clf.n_classes_ = 2
    return clf


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
    """
    Choose a threshold using the *empirical score distribution*.

    XGBoost probabilities are not guaranteed to be well-calibrated to [0,1] in a way that
    matches arbitrary fixed cutoffs; a grid on absolute [0.3, 0.9] can miss the entire
    live score range and yield 0 trades.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) == 0:
        return 0.60

    qs = np.linspace(0.50, 0.95, 19)  # quantiles of predicted scores
    thresh_candidates = np.unique(np.quantile(y_prob, qs))

    best_t = float(np.quantile(y_prob, 0.80))
    best_score = -1e9

    for t in thresh_candidates:
        y_pred = (y_prob >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        buy_rate = float(y_pred.mean())
        realized = forward_returns[y_pred == 1] if y_pred.any() else np.array([])
        avg_trade_return = float(realized.mean()) if realized.size else -0.01
        expectancy = avg_trade_return - (TRADE_COST_PCT / 100.0)

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

    return float(best_t)


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
        fold_model = _train_xgb_classifier(
            X_tr,
            y_tr,
            X_val,
            y_val,
            spw,
            random_state=42 + i,
            feature_names=FEATURES,
            num_boost_round=1500,
            early_stopping_rounds=100,
            max_depth=4,
            eta=0.03,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=2.5,
            reg_alpha=0.8,
            min_child_weight=5,
            gamma=0.15,
        )
        y_prob_val = fold_model.predict_proba(X_val)[:, 1]
        candidates.append(_find_best_threshold(y_val, y_prob_val, r_val))

    if not candidates:
        return 0.70
    return float(np.median(candidates))


def train_model():
    print("Collecting training data...")

    row_parts = []
    per_ticker_counts = {}
    failed_tickers = []

    for ticker in TRAIN_TICKERS:
        part = _build_dataset_for_ticker(ticker, days=None)
        if part:
            row_parts.append(part)
            per_ticker_counts[ticker] = len(part)
        else:
            failed_tickers.append(ticker)

    X, y, r = _merge_time_sorted_rows(row_parts)

    if len(X) == 0:
        raise RuntimeError(
            "No real historical training data loaded from Alpaca. "
            "Synthetic fallback is disabled. Check credentials/data access."
        )

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

    model = _train_xgb_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        scale_pos_weight,
        random_state=42,
        feature_names=FEATURES,
        num_boost_round=3000,
        early_stopping_rounds=200,
        max_depth=4,
        eta=0.025,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_lambda=2.5,
        reg_alpha=0.8,
        min_child_weight=5,
        gamma=0.15,
    )

    # Tune threshold ONLY on validation set (no test leakage).
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_auc = (
        roc_auc_score(y_val, y_prob_val) if len(np.unique(y_val)) > 1 else 0.5
    )
    val_threshold = _find_best_threshold(y_val, y_prob_val, r_val)
    best_threshold = float(np.clip((walk_forward_threshold + val_threshold) / 2.0, 0.05, 0.99))

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
    print(
        f"Metrics: acc={acc:.3f} precision={precision:.3f} recall={recall:.3f} "
        f"f1={f1:.3f} val_auc={val_auc:.3f} test_auc={auc:.3f}"
    )
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
            "val_auc": float(val_auc),
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
