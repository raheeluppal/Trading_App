"""
Train XGBoost model using only real historical market data.
Fails fast if live historical data is unavailable.
"""

import argparse
import json
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score

from features import build_feature_matrix, MODEL_FEATURE_COLUMNS
from ranking_metrics import ranking_edge_report

warnings.filterwarnings("ignore")

FEATURES_BASE = list(MODEL_FEATURE_COLUMNS)
FEATURES = FEATURES_BASE + ["ticker_idx"]

# Full universe: used for `ticker_idx` encoding (must match live inference).
# ETFs + mega-cap tech + sector/style + banks/energy/healthcare for cross-section.
TRAIN_TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "META",
    "GOOGL",
    "AMD",
    "XLF",
    "XLE",
    "JPM",
    "JNJ",
    "XOM",
]
# 10 trading days (~2 weeks): aligns better with SMA50 / 126d drawdown features than 5-day noise.
FORWARD_BARS = 10
RETURN_THRESHOLD = 0.012  # +1.2% over 10 sessions (fixed-label mode)
NEGATIVE_RETURN_THRESHOLD = -0.010  # -1.0% over 10 sessions
# Every N bars per ticker reduces overlapping forward windows in the training set (1 = off).
SAMPLE_STRIDE = 2
# Rows dropped between train|val|test so adjacent-split rows do not share overlapping label windows.
EMBARGO_ROWS = 10
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


def _resolve_run_tickers(run_tickers):
    """Subset of TRAIN_TICKERS to fetch and train on (default: all)."""
    if run_tickers is None:
        return list(TRAIN_TICKERS)
    out = []
    for t in run_tickers:
        u = str(t).strip().upper()
        if not u:
            continue
        if u not in TRAIN_TICKERS:
            raise ValueError(
                f"Ticker {u} is not in the training universe {TRAIN_TICKERS}. "
                "Add it to TRAIN_TICKERS (and model inference) first."
            )
        out.append(u)
    if not out:
        raise ValueError("No tickers to train after parsing run_tickers.")
    # Preserve universe order for stable runs
    return [t for t in TRAIN_TICKERS if t in set(out)]


def _build_dataset_for_ticker(
    ticker: str,
    days=None,
    *,
    label_mode: str = "quantile",
    quantile_warmup: int = 100,
    q_lower: float = 0.33,
    q_upper: float = 0.67,
    sample_stride: int = 1,
    spy_bars: Optional[pd.DataFrame] = None,
):
    """
    label_mode:
      - fixed: +RETURN_THRESHOLD vs NEGATIVE_RETURN_THRESHOLD (drops ambiguous zone).
      - quantile: expanding window on *past* forward returns only — top q_upper vs bottom q_lower
        (adapts to volatility regime; often better AUC than fixed % moves on indices).
    """
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
        # SPY rows: market return = SPY return (use same bars). Other names: align SPY index.
        use_spy = bars if ticker == "SPY" else spy_bars
        fm = build_feature_matrix(bars, spy_bars=use_spy)
    except Exception:
        return []
    mat = fm.to_numpy(dtype=float)
    forward_ret_ser = (bars["close"].shift(-FORWARD_BARS) - bars["close"]) / bars["close"]
    forward_ret = forward_ret_ser.to_numpy()
    rows = []
    n = len(bars)
    if label_mode == "fixed":
        loop_start = 60
    elif label_mode == "quantile":
        # Need `quantile_warmup` past realized forward returns (from bar 60 onward).
        loop_start = 60 + quantile_warmup
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    st = max(1, int(sample_stride))
    for i in range(loop_start, n - FORWARD_BARS):
        if (i - loop_start) % st != 0:
            continue
        fr = forward_ret[i]
        if not np.isfinite(fr):
            continue

        if label_mode == "fixed":
            if fr >= RETURN_THRESHOLD:
                label = 1
            elif fr <= NEGATIVE_RETURN_THRESHOLD:
                label = 0
            else:
                continue
        else:
            past = forward_ret[60:i]
            past = past[np.isfinite(past)]
            if len(past) < quantile_warmup:
                continue
            lo, hi = np.quantile(past, [q_lower, q_upper])
            if fr >= hi:
                label = 1
            elif fr <= lo:
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


def _sample_weights_from_returns(forward_returns: np.ndarray) -> np.ndarray:
    """
    Up-weight larger |forward return| so the booster focuses on economically meaningful moves.
    Capped so outliers do not dominate.
    """
    a = np.asarray(np.abs(forward_returns), dtype=float)
    # ~0.5% move -> w~0.75, ~2%+ -> w~1.5, cap 2.0
    w = 0.5 + np.clip(a / 0.02, 0.0, 1.5)
    return np.clip(w, 0.35, 2.0).astype(float)


def _time_split_3way_embargoed(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    embargo_rows: int = 10,
    min_test_rows: int = 60,
):
    """
    Chronological split with gaps between regions so label windows [t, t+H] do not straddle
    train/val or val/test boundaries (approximate purging for overlapping horizons).
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    e = max(0, int(embargo_rows))
    val_start = train_end + e
    test_start = val_end + e
    if (
        val_start >= val_end
        or test_start >= n
        or (n - test_start) < min_test_rows
        or (val_end - val_start) < min_test_rows
    ):
        return _time_split_3way(X, y, r, train_ratio=train_ratio, val_ratio=val_ratio)
    return (
        X[:train_end],
        X[val_start:val_end],
        X[test_start:],
        y[:train_end],
        y[val_start:val_end],
        y[test_start:],
        r[:train_end],
        r[val_start:val_end],
        r[test_start:],
    )


def _train_xgb_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    scale_pos_weight,
    *,
    random_state,
    feature_names=None,
    train_weight=None,
    num_boost_round=4000,
    early_stopping_rounds=250,
    max_depth=5,
    eta=0.03,
    subsample=0.82,
    colsample_bytree=0.82,
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
        "max_delta_step": 1,
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": "hist",
        "seed": int(random_state),
    }
    fn = list(feature_names) if feature_names is not None else None
    tw = train_weight
    if tw is not None:
        tw = np.asarray(tw, dtype=float)
        if len(tw) != len(X_train):
            raise ValueError("train_weight length must match X_train")
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=tw, feature_names=fn)
    else:
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


def _walk_forward_threshold(
    X_train: np.ndarray, y_train: np.ndarray, r_train: np.ndarray, folds: int = 4
) -> Optional[float]:
    n = len(X_train)
    fold_size = n // (folds + 1)
    candidates = []
    for i in range(1, folds + 1):
        train_end = fold_size * i
        val_end = fold_size * (i + 1)
        # Allow smaller training folds (e.g. single-ticker ~1k rows) — 300 was too strict.
        if val_end > n or train_end < 120:
            break

        X_tr, y_tr = X_train[:train_end], y_train[:train_end]
        X_val, y_val = X_train[train_end:val_end], y_train[train_end:val_end]
        r_val = r_train[train_end:val_end]
        if len(X_val) < 80 or len(np.unique(y_tr)) < 2:
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
            num_boost_round=2000,
            early_stopping_rounds=120,
            max_depth=5,
            eta=0.03,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_lambda=2.0,
            reg_alpha=0.55,
            min_child_weight=3,
            gamma=0.12,
        )
        y_prob_val = fold_model.predict_proba(X_val)[:, 1]
        candidates.append(_find_best_threshold(y_val, y_prob_val, r_val))

    if not candidates:
        return None
    return float(np.median(candidates))


def _align_walk_forward_to_val_scores(
    walk_forward: Optional[float],
    y_prob_val: np.ndarray,
) -> Optional[float]:
    """
    Fold models can produce threshold candidates outside the main model's score band
    on the same validation set (e.g. wf=0.615 while val scores top out ~0.57). Clip wf
    to the bulk of y_prob_val before blending.
    """
    if walk_forward is None:
        return None
    y = np.asarray(y_prob_val, dtype=float)
    if len(y) < 10:
        return float(walk_forward)
    lo, hi = float(np.quantile(y, 0.02)), float(np.quantile(y, 0.995))
    return float(np.clip(walk_forward, lo, hi))


def _blend_decision_thresholds(
    walk_forward: Optional[float], validation_threshold: float
) -> float:
    """Do not blend with a fake default — that inflated thresholds and killed all BUYs."""
    if walk_forward is None:
        return float(validation_threshold)
    return float(np.clip((walk_forward + validation_threshold) / 2.0, 0.05, 0.99))


def _cap_threshold_for_min_top_frac(
    y_prob: np.ndarray, threshold: float, min_top_frac: float = 0.02
) -> float:
    """
    If threshold is above almost all scores, lower it so at least min_top_frac
    of the set scores at or above threshold (validation only — avoids saved threshold
    that yields zero trades when score distribution is tight).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) == 0:
        return threshold
    k = max(1, int(np.ceil(min_top_frac * len(y_prob))))
    k = min(k, len(y_prob))
    # k-th largest score: at most k-1 values are strictly greater
    t_floor = float(np.partition(y_prob, -k)[-k])
    return float(min(threshold, t_floor))


def train_model(
    run_tickers=None,
    *,
    label_mode: str = "quantile",
    quantile_warmup: int = 100,
    q_lower: float = 0.28,
    q_upper: float = 0.72,
    sample_stride: int = SAMPLE_STRIDE,
    embargo_rows: int = EMBARGO_ROWS,
):
    """
    Train the classifier. Pass run_tickers=['SPY'] (or use --tickers SPY) for a fast
    single-symbol experiment. `ticker_idx` still uses the full TRAIN_TICKERS universe.

    label_mode:
      - quantile (default): labels from expanding distribution of past forward returns
        (often clearer signal than fixed % thresholds on indices).
      - fixed: use RETURN_THRESHOLD / NEGATIVE_RETURN_THRESHOLD only.
    """
    tickers_to_run = _resolve_run_tickers(run_tickers)
    lm = str(label_mode).lower().strip()
    if lm not in ("fixed", "quantile"):
        raise ValueError("label_mode must be 'fixed' or 'quantile'")

    print("Collecting training data...")
    print(f"Symbols this run: {tickers_to_run} (ticker_idx universe: {len(TRAIN_TICKERS)} names)")
    print(f"Label mode: {lm}" + (f" (warmup={quantile_warmup}, q=[{q_lower},{q_upper}])" if lm == "quantile" else ""))
    print(
        f"Horizon={FORWARD_BARS} bars | sample_stride={sample_stride} (decorrelate labels) | "
        f"split_embargo={embargo_rows} rows between train/val/test"
    )

    spy_bars_global = None
    try:
        from data_feed import get_historical_bars_for_training as _gf_spy

        spr = _gf_spy("SPY", days=None)
        spy_bars_global = _normalize_bars_df(spr)
        if spy_bars_global is None or spy_bars_global.empty:
            spy_bars_global = None
            print("⚠ SPY history missing — mkt_ret_* features will be 0 for non-SPY names.")
        else:
            print(f"✓ SPY loaded for market features ({len(spy_bars_global)} bars)")
    except Exception as ex:
        print(f"⚠ SPY load failed ({ex}); mkt_ret_* = 0 for non-SPY names.")
        spy_bars_global = None

    row_parts = []
    per_ticker_counts = {}
    failed_tickers = []

    for ticker in tickers_to_run:
        part = _build_dataset_for_ticker(
            ticker,
            days=None,
            label_mode=lm,
            quantile_warmup=quantile_warmup,
            q_lower=q_lower,
            q_upper=q_upper,
            sample_stride=sample_stride,
            spy_bars=spy_bars_global,
        )
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
    ) = _time_split_3way_embargoed(
        X, y, r, train_ratio=0.7, val_ratio=0.15, embargo_rows=EMBARGO_ROWS
    )

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training labels are single-class; cannot train classifier.")
    if len(X_val) < 60 or len(X_test) < 60:
        raise RuntimeError(
            "Insufficient validation/test samples (need at least 60 each). "
            "Train on more tickers or use daily history with more bars."
        )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = max(1.0, neg / max(pos, 1))

    walk_forward_threshold = _walk_forward_threshold(X_train, y_train, r_train, folds=4)
    walk_forward_raw = walk_forward_threshold

    # Slightly less aggressive regularization when labels are quantile-based (clearer class gap).
    reg_boost = 1.0 if lm == "fixed" else 0.88
    sw_train = _sample_weights_from_returns(r_train)
    model = _train_xgb_classifier(
        X_train,
        y_train,
        X_val,
        y_val,
        scale_pos_weight,
        random_state=42,
        feature_names=FEATURES,
        train_weight=sw_train,
        num_boost_round=4000,
        early_stopping_rounds=250,
        max_depth=5,
        eta=0.03,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=2.0 * reg_boost,
        reg_alpha=0.55 * reg_boost,
        min_child_weight=max(2, int(round(3 * reg_boost))),
        gamma=0.12 * reg_boost,
    )

    # Tune threshold ONLY on validation set (no test leakage).
    y_prob_val = model.predict_proba(X_val)[:, 1]
    val_auc = (
        roc_auc_score(y_val, y_prob_val) if len(np.unique(y_val)) > 1 else 0.5
    )
    walk_forward_threshold = _align_walk_forward_to_val_scores(
        walk_forward_threshold, y_prob_val
    )
    val_threshold = _find_best_threshold(y_val, y_prob_val, r_val)
    best_threshold = _blend_decision_thresholds(walk_forward_threshold, val_threshold)
    best_threshold = _cap_threshold_for_min_top_frac(y_prob_val, best_threshold, min_top_frac=0.02)

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
    net_expectancy_test = (
        avg_trade_return - (TRADE_COST_PCT / 100.0) if selected_returns.size else None
    )

    rank_rep = ranking_edge_report(
        y_prob_test,
        r_test,
        top_frac=0.10,
        cost_fraction=TRADE_COST_PCT / 100.0,
        n_perm=3000,
        n_bootstrap=2000,
        seed=42,
    )

    print("Training complete.")
    print(f"Samples: total={len(X)} train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"Class balance (train): pos={pos} neg={neg} scale_pos_weight={scale_pos_weight:.2f}")
    wf_s = f"{walk_forward_threshold:.3f}" if walk_forward_threshold is not None else "n/a"
    if (
        walk_forward_raw is not None
        and walk_forward_threshold is not None
        and abs(walk_forward_raw - walk_forward_threshold) > 1e-4
    ):
        print(
            f"Walk-forward threshold aligned to val score band: "
            f"{walk_forward_raw:.3f} -> {walk_forward_threshold:.3f}"
        )
    print(
        f"Score range: val=[{y_prob_val.min():.3f},{y_prob_val.max():.3f}] "
        f"test=[{y_prob_test.min():.3f},{y_prob_test.max():.3f}]"
    )
    print(
        f"Metrics: acc={acc:.3f} precision={precision:.3f} recall={recall:.3f} "
        f"f1={f1:.3f} val_auc={val_auc:.3f} test_auc={auc:.3f}"
    )
    print(
        f"Thresholds: walk_forward={wf_s} "
        f"validation={val_threshold:.3f} final={best_threshold:.3f}"
    )
    print(f"Predicted BUY rate on holdout: {buy_rate:.3f}")
    print(f"Avg selected forward return (gross): {avg_trade_return*100:.3f}%")
    if net_expectancy_test is not None:
        print(
            f"Estimated net expectancy on test BUYs (after ~{TRADE_COST_PCT}% cost): "
            f"{net_expectancy_test*100:.3f}%"
        )
    if auc < 0.55:
        print(
            "⚠ test_auc < 0.55: treat as no proven edge — paper-trade before risking capital."
        )
    if rank_rep.get("spearman_ic") is not None:
        lf = rank_rep.get("lift_mean_fwd")
        lo = rank_rep.get("lift_bootstrap_ci95_low")
        hi = rank_rep.get("lift_bootstrap_ci95_high")
        lift_s = (
            f"top10% lift={lf*100:.3f}% [bootstrap 95% CI {lo*100:.3f}%, {hi*100:.3f}%]"
            if lf is not None and lo is not None and hi is not None
            else "top10% lift=n/a"
        )
        print(
            f"Ranking (test): Spearman IC={rank_rep['spearman_ic']:.4f} "
            f"(perm p≈{rank_rep.get('spearman_ic_pvalue_perm', 1.0):.4f}) | "
            f"{lift_s}"
        )
    print(f"Per-ticker sample counts: {per_ticker_counts}")
    if failed_tickers:
        print(f"Tickers with insufficient/unavailable history: {failed_tickers}")

    model.get_booster().save_model(str(MODEL_PATH))

    metadata = {
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "decision_threshold": best_threshold,
        "label_mode": lm,
        "quantile_warmup": int(quantile_warmup) if lm == "quantile" else None,
        "quantile_lower": float(q_lower) if lm == "quantile" else None,
        "quantile_upper": float(q_upper) if lm == "quantile" else None,
        "forward_bars": FORWARD_BARS,
        "sample_stride": int(sample_stride),
        "embargo_rows": int(embargo_rows),
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
            "test_net_expectancy_after_cost_est": net_expectancy_test,
            "ranking_validation": rank_rep,
        },
        "walk_forward_threshold": walk_forward_threshold,
        "walk_forward_threshold_raw": walk_forward_raw,
        "split": {
            "train_size": int(len(X_train)),
            "validation_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "train_tickers": tickers_to_run,
        "ticker_idx_universe": list(TRAIN_TICKERS),
        "per_ticker_counts": per_ticker_counts,
        "failed_tickers": failed_tickers,
    }

    with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {MODEL_META_PATH}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost on historical daily bars (Alpaca)."
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated symbols to train on (subset of built-in universe), e.g. SPY for a quick run.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("quantile", "fixed"),
        default="quantile",
        help="quantile (default): top/bottom of expanding past forward-return distribution. "
        "fixed: use fixed %% move thresholds from RETURN_THRESHOLD settings.",
    )
    parser.add_argument("--quantile-warmup", type=int, default=100, help="Bars of history for quantile labels.")
    parser.add_argument(
        "--q-lower",
        type=float,
        default=0.28,
        dest="q_lower",
        help="Lower quantile for negative class (more extreme = sharper labels).",
    )
    parser.add_argument(
        "--q-upper",
        type=float,
        default=0.72,
        dest="q_upper",
        help="Upper quantile for positive class.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=SAMPLE_STRIDE,
        dest="sample_stride",
        help="Use every Nth bar per ticker to reduce overlapping forward-return labels (default 2).",
    )
    parser.add_argument(
        "--embargo-rows",
        type=int,
        default=EMBARGO_ROWS,
        dest="embargo_rows",
        help="Gap rows between train/val and val/test splits (default 10, ~horizon).",
    )
    args = parser.parse_args()
    run = None
    if args.tickers:
        run = [t.strip() for t in args.tickers.split(",") if t.strip()]
    train_model(
        run_tickers=run,
        label_mode=args.label_mode,
        quantile_warmup=args.quantile_warmup,
        q_lower=args.q_lower,
        q_upper=args.q_upper,
        sample_stride=max(1, args.sample_stride),
        embargo_rows=max(0, args.embargo_rows),
    )
