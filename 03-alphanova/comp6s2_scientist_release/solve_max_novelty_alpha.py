"""Directly optimize feature weights to maximize min distance to database while Net Sharpe >= 0.015."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from city_tools import compute_gauge_fixed, compute_city
from evaluation import backtest

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
df = pd.read_parquet(city_db_path)
dim_cols = [f"city_dim_{i}" for i in range(19)]
db_cities = df[dim_cols].values
db_norms = np.linalg.norm(db_cities, axis=1, keepdims=True)
db_cities = db_cities / np.maximum(db_norms, 1e-8)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

# Building blocks: 8 orthogonal components
component_names = [
    "inv_core",   # -(F1..F4)/4
    "f6_f1",      # F6 - F1
    "f3_f5",      # F3 - F5
    "f5_f4",      # F5 - F4
    "f2_f5",      # F2 - F5
    "inv_f6",     # -F6
    "cubic_core", # -(F1..F4)^3 / 4
    "ratio_6_1",  # F6 / (|F1|+0.1)
]

def get_components(g, t):
    c = {}
    c["inv_core"] = -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0
    c["f6_f1"] = csrank(g[6][t] - g[1][t])
    c["f3_f5"] = csrank(g[3][t] - g[5][t])
    c["f5_f4"] = csrank(g[5][t] - g[4][t])
    c["f2_f5"] = csrank(g[2][t] - g[5][t])
    c["inv_f6"] = -csrank(g[6][t])
    c["cubic_core"] = -(csrank(g[1][t])**3 + csrank(g[2][t])**3 + csrank(g[3][t])**3 + csrank(g[4][t])**3) / 4.0
    c["ratio_6_1"] = csrank(g[6][t] / (g[1][t].abs() + 0.1))
    return c

# Precompute component predictions across all periods
comp_cache = {name: [] for name in component_names}
all_ret = []
all_tgt = []
all_tickers = []
all_gauge = []

for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
    g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}
    all_tickers.append(tickers)
    all_ret.append(ret[tickers])
    all_tgt.append(tgt[tickers])
    all_gauge.append(harness._cached_gauge[idx])

    comps = get_components(g, tickers)
    for name in component_names:
        comp_cache[name].append(comps[name])

print("Precomputed components. Running optimization...")

def eval_w(w, span=220):
    all_pnl = []
    all_ic = []
    all_gf = []

    for idx in range(len(harness._cached_val_data)):
        tickers = all_tickers[idx]
        ret = all_ret[idx]
        tgt = all_tgt[idx]
        g_mat = all_gauge[idx]

        sig = sum(w[i] * comp_cache[name][idx] for i, name in enumerate(component_names))
        sig = sig.sub(sig.mean(axis=1), axis=0)

        smoothed = sig.ewm(span=span, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret)
        all_pnl.append(pnl)

        dot = (smoothed * tgt).sum(axis=1)
        norm_s = np.sqrt((smoothed**2).sum(axis=1))
        norm_t = np.sqrt((tgt**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        if g_mat is not None:
            p_vals = smoothed.values - smoothed.values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(p_vals))
            all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)

    dots = db_cities @ city
    min_dist_deg = np.degrees(np.arccos(np.clip(np.max(dots), -1, 1)))
    return net_sh, np.mean(all_ic), min_dist_deg, city

# Grid search across different novel spaces
proposals = [
    ("Cubic Core + F3_F5 Spread", [0.0, 0.20, 0.40, 0.30, 0.10, 0.0, 0.40, 0.0]),
    ("Cubic Core + F6_F1 Spread", [0.0, 0.35, 0.25, 0.20, 0.0, 0.0, 0.45, 0.0]),
    ("F3_F5 & F5_F4 dominant + Cubic", [0.0, 0.15, 0.45, 0.35, 0.15, 0.10, 0.30, 0.0]),
    ("Relative Value + Ratio + Cubic", [0.0, 0.20, 0.35, 0.30, 0.10, 0.10, 0.25, 0.20]),
    ("Pure Cubic Tail + Spreads", [0.0, 0.25, 0.30, 0.25, 0.10, 0.0, 0.50, 0.0]),
    ("Balanced Multi-Spread Cubic", [0.10, 0.20, 0.35, 0.30, 0.10, 0.10, 0.25, 0.10]),
]

print("=" * 85)
print(f"{'Proposal Name':<35} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Min DB Dist':<13}")
print("=" * 85)

for name, w in proposals:
    sh, ic, dist, _ = eval_w(w, span=220)
    print(f"{name:<35} | {sh:>+10.4f} | {ic:>+8.4f} | {dist:>11.1f}°")
