"""Tune R2 Archetype weights to maximize Sharpe and distance from Attempt 4."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, check_cross_sectional_demean
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(axis=1).values.reshape(-1, 1)
    s = df.std(axis=1).values.reshape(-1, 1) + 1e-8
    return (df - m) / s

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

# Compute Attempt 4 city
all_gf_att4 = []
for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    tickers = sorted(list(item["val_returns"].columns), key=lambda c: int(c.split(".")[-1]))
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 5)}
    sig = -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
    sig = sig.ewm(span=150, min_periods=1).mean().sub(sig.mean(axis=1), axis=0)
    g_mat = harness._cached_gauge[idx]
    if g_mat is not None:
        p_vals = sig[tickers].values - sig[tickers].values.mean(axis=1, keepdims=True)
        n_use = min(len(g_mat), len(p_vals))
        all_gf_att4.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

c_att4 = compute_city(np.concatenate(all_gf_att4, axis=0))

# Precompute building blocks across all periods
blocks = {
    "s35": [],
    "s54": [],
    "s25": [],
    "inv_f6": [],
    "inv_core": [],
    "inv_f1": [],
    "inv_f4": [],
    "ratio": [],
    "f6_f1": [],
}

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
    blocks["s35"].append(csrank(g[3] - g[5]))
    blocks["s54"].append(csrank(g[5] - g[4]))
    blocks["s25"].append(csrank(g[2] - g[5]))
    blocks["inv_f6"].append(-csrank(g[6]))
    blocks["inv_core"].append(-(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0)
    blocks["inv_f1"].append(-csrank(g[1]))
    blocks["inv_f4"].append(-csrank(g[4]))
    blocks["ratio"].append(csrank(g[6] / ((g[1] + g[3]).abs() + 0.1)))
    blocks["f6_f1"].append(csrank(g[6] - g[1]))

print("Precomputed blocks. Testing weight variants...")

def test_config(weights, span=220):
    all_pnl = []
    all_ic = []
    all_gf = []

    for idx in range(len(harness._cached_val_data)):
        tickers = all_tickers[idx]
        ret = all_ret[idx]
        tgt = all_tgt[idx]

        sig = sum(w * blocks[name][idx] for name, w in weights.items())
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

        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_vals = smoothed.values - smoothed.values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(p_vals))
            all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)
    comp_df = compare_to_cities(city, city_db_path=city_db_path)
    min_dist = comp_df["angular_distance_deg"].iloc[0]
    nearest = comp_df["signal_id"].iloc[0]

    d_att4 = np.degrees(np.arccos(np.clip(np.dot(city, c_att4), -1, 1)))
    return net_sh, np.mean(all_ic), d_att4, min_dist, nearest

variants = [
    ("Base R2 Archetype", {"s35": 0.35, "s54": 0.30, "inv_f6": 0.20, "s25": 0.15, "inv_f4": 0.15, "ratio": 0.10, "inv_core": 0.0, "inv_f1": 0.0, "f6_f1": 0.0}),
    ("R2 + 0.15 Core", {"s35": 0.35, "s54": 0.30, "inv_f6": 0.20, "s25": 0.15, "inv_f4": 0.15, "ratio": 0.10, "inv_core": 0.15, "inv_f1": 0.0, "f6_f1": 0.0}),
    ("R2 + 0.25 Core", {"s35": 0.30, "s54": 0.25, "inv_f6": 0.15, "s25": 0.15, "inv_f4": 0.10, "ratio": 0.10, "inv_core": 0.25, "inv_f1": 0.0, "f6_f1": 0.0}),
    ("R2 + 0.35 Core", {"s35": 0.25, "s54": 0.20, "inv_f6": 0.15, "s25": 0.10, "inv_f4": 0.10, "ratio": 0.10, "inv_core": 0.35, "inv_f1": 0.0, "f6_f1": 0.0}),
    ("R2 + Core + F6_F1 Spread", {"s35": 0.30, "s54": 0.25, "inv_f6": 0.15, "s25": 0.15, "inv_f4": 0.10, "ratio": 0.10, "inv_core": 0.20, "inv_f1": 0.0, "f6_f1": 0.20}),
    ("Spreads 3-5 & 5-4 dominant + Core", {"s35": 0.40, "s54": 0.35, "inv_f6": 0.10, "s25": 0.10, "inv_f4": 0.0, "ratio": 0.0, "inv_core": 0.25, "inv_f1": 0.0, "f6_f1": 0.15}),
    ("Pure Spreads + F6_F1 (Zero Core)", {"s35": 0.45, "s54": 0.35, "inv_f6": 0.20, "s25": 0.15, "inv_f4": 0.0, "ratio": 0.10, "inv_core": 0.0, "inv_f1": 0.0, "f6_f1": 0.25}),
]

print("=" * 115)
print(f"{'Variant':<35} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Dist to Att#4':<13} | {'Min Dist':<9} | {'Nearest Signal':<20}")
print("=" * 115)

for name, w in variants:
    sh, ic, d4, md, nid = test_config(w, span=220)
    print(f"{name:<35} | {sh:>+10.4f} | {ic:>+8.4f} | {d4:>11.1f}° | {md:>7.1f}° | {nid:<20}")
