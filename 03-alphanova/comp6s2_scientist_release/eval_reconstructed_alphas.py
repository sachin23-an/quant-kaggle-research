"""Evaluate walk-forward performance of reconstructed Rank 2 and Rank 3 alphas."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
df = pd.read_parquet(city_db_path)
dim_cols = [f"city_dim_{i}" for i in range(19)]
r2_city = df[df["predictor_id"] == "signal_1d4528f4db1c33c1"][dim_cols].values[0]
r3_city = df[df["predictor_id"] == "signal_9ebffdb75b09eb12"][dim_cols].values[0]

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

# 1. Formulation based on Rank 2 weights:
def sig_r2_formula(g, tickers):
    # Dominant features from regression:
    # spread_3_5 (+), spread_2_5 (+), z_F3 (+), -spread_4_5 (+ F5 - F4), -z_F4, -rank_F6
    s35 = csrank(g[3] - g[5])
    s25 = csrank(g[2] - g[5])
    s54 = csrank(g[5] - g[4])
    inv_f6 = -csrank(g[6])
    inv_f4 = -zscore(g[4])
    ratio = csrank(g[6] / ((g[1] + g[3]).abs() + 0.1))
    
    raw = 0.35 * s35 + 0.30 * s54 + 0.20 * inv_f6 + 0.15 * s25 + 0.15 * inv_f4 + 0.10 * ratio
    return raw

# 2. Formulation based on Rank 3 weights:
def sig_r3_formula(g, tickers):
    s35 = csrank(g[3] - g[5])
    s54 = csrank(g[5] - g[4])
    inv_f6 = -csrank(g[6])
    ratio = csrank(g[6] / ((g[1] + g[3]).abs() + 0.1))
    
    raw = 0.40 * s35 + 0.35 * s54 + 0.30 * inv_f6 + 0.20 * ratio
    return raw

# 3. Clean Relative Spread Inversion:
def sig_relative_spread_alpha(g, tickers):
    # F3 - F5 spread + F5 - F4 spread = F3 - F4!
    # With F6 relative value
    return csrank(g[3] - g[5]) + csrank(g[5] - g[4]) - 0.4 * csrank(g[6])

candidates = {
    "R2 Reconstructed Alpha": sig_r2_formula,
    "R3 Reconstructed Alpha": sig_r3_formula,
    "Relative Spread Inversion (3-5 + 5-4 - F6)": sig_relative_spread_alpha,
}

print("=" * 115)
print(f"{'Alpha Formulation':<42} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Dist to Att#4':<13} | {'Min Dist':<9} | {'Nearest Signal':<20}")
print("=" * 115)

for name, func in candidates.items():
    all_pnl = []
    all_ic = []
    all_gf = []

    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tgt = item["val_target"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}

        sig = func(g, tickers)
        sig = sig.sub(sig.mean(axis=1), axis=0)

        smoothed = sig.ewm(span=200, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret[tickers])
        all_pnl.append(pnl)

        dot = (smoothed * tgt[tickers]).sum(axis=1)
        norm_s = np.sqrt((smoothed**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_vals = smoothed[tickers].values - smoothed[tickers].values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(p_vals))
            all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)

    comp_df = compare_to_cities(city, city_db_path=city_db_path)
    min_comp_dist = comp_df["angular_distance_deg"].iloc[0]
    nearest_sig = comp_df["signal_id"].iloc[0]

    d_att4 = np.degrees(np.arccos(np.clip(np.dot(city, c_att4), -1, 1)))

    print(f"{name:<42} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | {d_att4:>11.1f}° | {min_comp_dist:>7.1f}° | {nearest_sig:<20}")
