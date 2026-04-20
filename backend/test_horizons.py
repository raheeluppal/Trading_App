"""
Test different prediction horizons to find optimal AUC.

Uses label_mode='quantile' (same as default train_model) — only FORWARD_BARS changes.

(Optional) Set HORIZON_TEST_TICKERS=SPY for a faster smoke run.

Each full train overwrites trained_model.json / trained_model_meta.json (last horizon wins).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def test_horizon(
    forward_bars: int,
    *,
    run_tickers: list[str] | None = None,
) -> None:
    """Train with quantile labels; override horizon via module global."""
    if "train_model" in sys.modules:
        del sys.modules["train_model"]

    import train_model

    train_model.FORWARD_BARS = int(forward_bars)

    print(f"\n{'=' * 70}")
    print(f"Testing FORWARD_BARS={forward_bars} bars (QUANTILE labels)")
    print(f"{'=' * 70}\n")

    try:
        train_model.train_model(
            run_tickers=run_tickers,
            label_mode="quantile",
            quantile_warmup=100,
            q_lower=0.28,
            q_upper=0.72,
            sample_stride=2,
            embargo_rows=10,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()


def main() -> None:
    raw = os.environ.get("HORIZON_TEST_TICKERS", "").strip()
    run_tickers = [t.strip() for t in raw.split(",") if t.strip()] if raw else None
    if run_tickers:
        print(f"HORIZON_TEST_TICKERS: {run_tickers}")

    configs = [1, 2, 3, 5, 10]

    print("\n" + "=" * 70)
    print("HORIZON COMPARISON TEST (QUANTILE LABELS)")
    print("=" * 70)

    for forward_bars in configs:
        test_horizon(forward_bars, run_tickers=run_tickers)

    print("\n" + "=" * 70)
    print("Test complete. Compare test_auc values in the Metrics lines above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
