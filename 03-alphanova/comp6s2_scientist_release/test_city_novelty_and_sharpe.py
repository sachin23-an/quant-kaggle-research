"""Analyze city novelty and Sharpe for various candidate alphas."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest
from city_tools import compute_gauge_fixed, compute_city, compute_ic, compare_to_cities

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

candidates = {
    "1. Pure -(F1..F4)/4": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0,
    "2. Pure -F1": lambda g, t: -csrank(g[1][t]),
    "3. Pure -(F1+F3)/2": lambda g, t: -(csrank(g[1][t]) + csrank(g[3][t])) / 2.0,
    "4. -(F1..F4)/4 + 0.15*(F6-F1)": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 + 0.15 * csrank(g[6][t] - g[1][t]),
    "5. -(F1..F4)/4 + 0.25*(F6-F2)": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 + 0.25 * csrank(g[6][t] - g[2][t]),
    "6. -(F1..F4)/4 + 0.3*F6": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 + 0.3 * csrank(g[6][t]),
    "7. Weighted Core -0.4*F1 - 0.3*F3 - 0.2*F4 - 0.1*F2": lambda g, t: -(0.4*csrank(g[1][t]) + 0.1*csrank(g[2][t]) + 0.3*csrank(g[3][t]) + 0.2*csrank(g[4][t])),
    "8. -(F1..F4)/4 + 0.2*(F6 - (F1+F2)/2)": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 + 0.2 * (csrank(g[6][t] - g[1][t]) + csrank(g[6][t] - g[2][t])) / 2.0,
    "9. -(F1..F4)/4 + 0.2*F6 - 0.1*F5": lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 + 0.2 * csrank(g[6][t]) - 0.1 * csrank(g[5][t]),
}

print("=" * 105)
print(f"{'Alpha Candidate':<42} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Min City Dist':<13} | {'Nearest Signal':<20}")
print("=" * 105)

for name, func in candidates.items():
    all_ic = []
    all_pnl = []
    all_gf = []

    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tgt = item["val_target"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}

        sig = func(g, tickers)
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
        pnl = backtest(smoothed, ret[tickers])
        all_pnl.append(pnl)

        # Gauge fix
        if harness._cached_gauge[idx] is not None:
            g_mat = harness._cached_gauge[idx]
            pred_vals = smoothed[tickers].values
            pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(pred_vals))
            gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
            all_gf.append(gf)

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)
    comp_df = compare_to_cities(city, city_db_path=city_db_path)
    min_dist_deg = comp_df["angular_distance_deg"].iloc[0]
    nearest_id = comp_df["signal_id"].iloc[0]

    print(f"{name:<42} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | {min_dist_deg:>10.2f}° | {nearest_id:<20}")
