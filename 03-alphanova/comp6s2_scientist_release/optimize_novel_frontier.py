"""Optimize linear weights across features and spreads to maximize Sharpe while keeping distance > 55 deg."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

# Precompute ranked factor blocks across all 31 validation periods
factor_names = ["inv_f1", "inv_f2", "inv_f3", "inv_f4", "f6", "f6_f1", "f6_f2", "f6_f3", "f6_f5", "inv_f5"]
factor_data = {fn: [] for fn in factor_names}
all_ret = []
all_tgt = []
all_tickers = []

for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
    all_tickers.append(tickers)
    all_ret.append(ret[tickers])
    all_tgt.append(tgt[tickers])

    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
    factor_data["inv_f1"].append(-csrank(g[1]))
    factor_data["inv_f2"].append(-csrank(g[2]))
    factor_data["inv_f3"].append(-csrank(g[3]))
    factor_data["inv_f4"].append(-csrank(g[4]))
    factor_data["f6"].append(csrank(g[6]))
    factor_data["f6_f1"].append(csrank(g[6] - g[1]))
    factor_data["f6_f2"].append(csrank(g[6] - g[2]))
    factor_data["f6_f3"].append(csrank(g[6] - g[3]))
    factor_data["f6_f5"].append(csrank(g[6] - g[5]))
    factor_data["inv_f5"].append(-csrank(g[5]))

print("Precomputed factor blocks. Testing weight vectors...")

def evaluate_weights(weights, span=200):
    all_pnl = []
    all_ic = []
    all_gf = []

    for idx in range(len(harness._cached_val_data)):
        tickers = all_tickers[idx]
        ret = all_ret[idx]
        tgt = all_tgt[idx]

        sig = sum(w * factor_data[fn][idx] for w, fn in zip(weights, factor_names))
        sig = sig.sub(sig.mean(axis=1), axis=0)

        # IC
        dot = (sig * tgt).sum(axis=1)
        norm_s = np.sqrt((sig**2).sum(axis=1))
        norm_t = np.sqrt((tgt**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        # Sharpe
        smoothed = sig.ewm(span=span, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret)
        all_pnl.append(pnl)

        # Gauge
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            pred_vals = smoothed.values
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

    return net_sh, np.mean(all_ic), min_dist_deg, nearest_id

# Test a grid of diverse weight vectors
weight_proposals = [
    ("Pure Core", [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0]),
    ("Core + F6_F1 Spread", [0.20, 0.20, 0.20, 0.20, 0, 0.30, 0, 0, 0, 0]),
    ("Core + F6_F2 Spread", [0.20, 0.20, 0.20, 0.20, 0, 0, 0.30, 0, 0, 0]),
    ("Core + F6_F3 Spread", [0.20, 0.20, 0.20, 0.20, 0, 0, 0, 0.30, 0, 0]),
    ("Core + F6_F5 Spread", [0.20, 0.20, 0.20, 0.20, 0, 0, 0, 0, 0.30, 0]),
    ("Core + F6 Overlay", [0.20, 0.20, 0.20, 0.20, 0.25, 0, 0, 0, 0, 0]),
    ("Core + F6 Overlay + Inv_F5", [0.20, 0.20, 0.20, 0.20, 0.25, 0, 0, 0, 0, 0.15]),
    ("F1+F3 dominant + F6 Spread", [0.35, 0.10, 0.35, 0.10, 0, 0.25, 0, 0, 0, 0]),
    ("F2+F4 dominant + F6 Spread", [0.10, 0.35, 0.10, 0.35, 0, 0, 0.25, 0, 0, 0]),
    ("Core + Multi-Spread Basket", [0.18, 0.18, 0.18, 0.18, 0.10, 0.15, 0.15, 0.15, 0.10, 0.05]),
    ("Relative Value Balanced", [0.15, 0.15, 0.15, 0.15, 0.15, 0.20, 0.10, 0.10, 0.10, 0.10]),
]

print("=" * 105)
print(f"{'Configuration':<35} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Min City Dist':<13} | {'Nearest Signal':<20}")
print("=" * 105)

for name, w in weight_proposals:
    sh, ic, dist, nid = evaluate_weights(w, span=220)
    print(f"{name:<35} | {sh:>+10.4f} | {ic:>+8.4f} | {dist:>10.2f}° | {nid:<20}")
