"""Accurate verification of raw and inverted alpha signals using official backtest() function."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest


def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def verify():
    harness = ResearchHarness()
    harness.preload_validation_cache(verbose=False)

    print("=" * 75)
    print("EXACT OFFICIAL BACKTEST OF CANDIDATE SIGNALS ACROSS ALL 31 PERIODS")
    print("=" * 75)

    spans = [60, 120, 150, 180, 240]

    # Test pure single feature inversions at various spans
    print(f"\n1. PURE INVERTED INDIVIDUAL FEATURES (-Fi):")
    print("-" * 75)
    print(f"{'Feature':<12} | {'span=60':<11} | {'span=120':<11} | {'span=150':<11} | {'span=180':<11} | {'span=240':<11}")
    print("-" * 75)

    for i in range(1, 7):
        row_str = f"Feature.{i:<4} |"
        for sp in spans:
            all_pnl = []
            for item in harness._cached_val_data:
                feat = item["val_features"][f"Feature.{i}"]
                ret = item["val_returns"]
                tickers = list(ret.columns)
                rf = csrank(feat[tickers])

                # Signal: -rf, smoothed
                sig = -rf.ewm(span=sp, min_periods=1).mean()
                sig = sig.sub(sig.mean(axis=1), axis=0)

                pnl = backtest(sig, ret)
                all_pnl.append(pnl)

            concat_pnl = pd.concat(all_pnl, ignore_index=True)
            sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0
            row_str += f" {sh:+10.4f} |"
        print(row_str)

    # Test combinations / baskets of inverted features
    print("\n2. BASKET COMBINATIONS:")
    print("-" * 75)
    baskets = {
        "-(F1+F3)/2": lambda f, t: -(csrank(f["Feature.1"][t]) + csrank(f["Feature.3"][t])) / 2.0,
        "-(F1+F2+F3+F4)/4": lambda f, t: -(csrank(f["Feature.1"][t]) + csrank(f["Feature.2"][t]) + csrank(f["Feature.3"][t]) + csrank(f["Feature.4"][t])) / 4.0,
        "-(F1+F4)/2": lambda f, t: -(csrank(f["Feature.1"][t]) + csrank(f["Feature.4"][t])) / 2.0,
        "-F1 only": lambda f, t: -csrank(f["Feature.1"][t]),
        "-F3 only": lambda f, t: -csrank(f["Feature.3"][t]),
        "-F4 only": lambda f, t: -csrank(f["Feature.4"][t]),
        "+F6 only (long)": lambda f, t: csrank(f["Feature.6"][t]),
        "-F6 only (short)": lambda f, t: -csrank(f["Feature.6"][t]),
    }

    for name, func in baskets.items():
        row_str = f"{name:<20} |"
        for sp in spans:
            all_pnl = []
            for item in harness._cached_val_data:
                feat = item["val_features"]
                ret = item["val_returns"]
                tickers = list(ret.columns)
                raw = func(feat, tickers)
                sig = raw.ewm(span=sp, min_periods=1).mean()
                sig = sig.sub(sig.mean(axis=1), axis=0)

                pnl = backtest(sig, ret)
                all_pnl.append(pnl)

            concat_pnl = pd.concat(all_pnl, ignore_index=True)
            sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0
            row_str += f" {sh:+10.4f} |"
        print(row_str)


if __name__ == "__main__":
    verify()
