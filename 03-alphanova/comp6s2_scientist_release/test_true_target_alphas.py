"""Measure true positive-target alpha combinations."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, TRANSACTION_COST
from city_tools import compute_gauge_fixed, compute_city, compute_ic


def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

print("=" * 80)
print("TESTING CANDIDATE ALPHA FORMULATIONS DIRECTLY AGAINST TARGET AND RETURNS")
print("=" * 80)

candidates = {
    "Pure -(F1..F4)/4": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0,
    "Pure -(F1+F3)/2": lambda g, t: -(csrank(g[1][t]) + csrank(g[3][t])) / 2.0,
    "Pure -F1": lambda g, t: -csrank(g[1][t]),
    "-(F1..F4)/4 with +F6 overlay": lambda g, t: (-(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0) + 0.3 * csrank(g[6][t]),
    "-(F1..F4)/4 with F6-Spreads": lambda g, t: (-(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0) + 0.2 * (csrank(g[6][t] - g[1][t]) + csrank(g[6][t] - g[2][t])) / 2.0,
    "F6 - Core": lambda g, t: csrank(g[6][t]) - (csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0,
}

for name, func in candidates.items():
    all_ic = []
    all_pnl = []
    all_gf = []

    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tgt = item["val_target"]
        tickers = list(ret.columns)
        g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}

        sig = func(g, tickers)
        # Demean
        sig = sig.sub(sig.mean(axis=1), axis=0)

        # IC vs Target
        dot = (sig * tgt[tickers]).sum(axis=1)
        norm_s = np.sqrt((sig**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        # Smooth for Sharpe
        smoothed = sig.ewm(span=240, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret)
        all_pnl.append(pnl)

        # Gauge fix
        if harness._cached_gauge[idx] is not None:
            g_mat = harness._cached_gauge[idx]
            pred_vals = smoothed.values
            cols = smoothed.columns.tolist()
            numeric_order = sorted(cols, key=lambda c: int(c.split(".")[-1]))
            if cols != numeric_order:
                pred_vals = smoothed[numeric_order].values
            pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(pred_vals))
            gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
            all_gf.append(gf)

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)
    dim0 = city[0]

    print(f"{name:<35} | Mean IC: {np.mean(all_ic):+.4f} | City Dim0: {dim0:+.4f} | Net Sharpe (span=240): {net_sh:+.4f}")
