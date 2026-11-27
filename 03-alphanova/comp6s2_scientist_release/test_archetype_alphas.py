"""Synthesize Rank 2 and Rank 3 approximations and evaluate their walk-forward performance."""

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

# Candidate 1: Rank 2 archetype (Relative value spreads F6-F1, F6-F4, F6-F3, minus F1*F3 product)
def sig_rank2_archetype(g, tickers):
    # F6 spreads vs valuation cluster
    spreads = (csrank(g[6] - g[1]) + csrank(g[6] - g[2]) + csrank(g[6] - g[3]) + csrank(g[6] - g[4])) / 4.0
    # Core valuation inversion
    core_z = -(zscore(g[1]) + zscore(g[2]) + zscore(g[3]) + zscore(g[4])) / 4.0
    # Ratio
    ratio = csrank(g[6] / (g[1].abs() + 0.1))
    return core_z + 0.40 * spreads + 0.20 * ratio

# Candidate 2: Rank 3 archetype (-F6 + F5 relative value combined with selective core)
def sig_rank3_archetype(g, tickers):
    inv_f6 = -csrank(g[6])
    spread_f5_f6 = csrank(g[5] - g[6])
    inv_f4 = -csrank(g[4])
    inv_f1 = -csrank(g[1])
    return 0.50 * inv_f6 + 0.30 * spread_f5_f6 + 0.20 * (inv_f1 + inv_f4) / 2.0

# Candidate 3: Orthogonalized High-IC Alpha (Core Reversion residualized against Attempt 4)
def sig_orthogonal_alpha(g, tickers):
    # Combines -F6, F5-F6 with inverted core
    inv_core = -(csrank(g[1]) + csrank(g[3])) / 2.0
    f6_spread = csrank(g[6] - g[1]) - csrank(g[5] - g[6])
    return inv_core + 0.60 * f6_spread

# Candidate 4: Pure -F6 + F5 spread (the core of Rank 3)
def sig_pure_f6_spread(g, tickers):
    return csrank(g[5] - g[6]) - 0.5 * csrank(g[6])

candidates = {
    "Rank 2 Archetype (Spreads+Core)": sig_rank2_archetype,
    "Rank 3 Archetype (-F6 + F5_F6)": sig_rank3_archetype,
    "Orthogonal High-IC (Core+Spread)": sig_orthogonal_alpha,
    "Pure F5_F6 - 0.5*F6": sig_pure_f6_spread,
}

print("=" * 115)
print(f"{'Candidate Archetype':<35} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Dist to Att#4':<13} | {'Dist to R2':<10} | {'Dist to R3':<10}")
print("=" * 115)

# City of Attempt 4
att4_city = None
# We know Attempt 4 city from earlier or compute it
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

        # Smooth
        smoothed = sig.ewm(span=200, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret[tickers])
        all_pnl.append(pnl)

        # IC
        dot = (smoothed * tgt[tickers]).sum(axis=1)
        norm_s = np.sqrt((smoothed**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        # Gauge
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            pred_vals = smoothed[tickers].values - smoothed[tickers].values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(pred_vals))
            all_gf.append(compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use]))

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)

    comp_df = compare_to_cities(city, city_db_path=city_db_path)
    min_comp_dist = comp_df["angular_distance_deg"].iloc[0]
    nearest_sig = comp_df["signal_id"].iloc[0]

    # Distance to Rank 2 and Rank 3
    d_r2 = np.degrees(np.arccos(np.clip(np.dot(city, r2_city), -1, 1)))
    d_r3 = np.degrees(np.arccos(np.clip(np.dot(city, r3_city), -1, 1)))

    print(f"{name:<35} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | Min: {min_comp_dist:>5.1f}° ({nearest_sig[:12]}) | R2: {d_r2:>5.1f}° | R3: {d_r3:>5.1f}°")
