"""Search for alphas that maximize distance from ALL database cities while keeping Sharpe > +0.015."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

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

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(axis=1).values.reshape(-1, 1)
    s = df.std(axis=1).values.reshape(-1, 1) + 1e-8
    return (df - m) / s

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

candidates = []

# Strategy A: Tail non-linearities (e.g. rank power 2, 3, or clipping)
for p in [1.5, 2.0, 2.5, 3.0]:
    candidates.append((
        f"Power Rank p={p}",
        lambda g, t, p=p: -(np.sign(csrank(g[1][t])) * (np.abs(csrank(g[1][t]))**p) + 
                            np.sign(csrank(g[2][t])) * (np.abs(csrank(g[2][t]))**p) + 
                            np.sign(csrank(g[3][t])) * (np.abs(csrank(g[3][t]))**p) + 
                            np.sign(csrank(g[4][t])) * (np.abs(csrank(g[4][t]))**p)) / 4.0
    ))

# Strategy B: Decile tail long/short (top decile long, bottom decile short)
for thresh in [0.4, 0.5, 0.6, 0.7]:
    candidates.append((
        f"Tail Truncation >{thresh}",
        lambda g, t, th=thresh: -(np.where(np.abs(csrank(g[1][t])) > th, csrank(g[1][t]), 0) +
                                 np.where(np.abs(csrank(g[2][t])) > th, csrank(g[2][t]), 0) +
                                 np.where(np.abs(csrank(g[3][t])) > th, csrank(g[3][t]), 0) +
                                 np.where(np.abs(csrank(g[4][t])) > th, csrank(g[4][t]), 0)) / 4.0
    ))

# Strategy C: Cross-feature dispersion gating
candidates.append((
    "Dispersion-Gated Core (high consensus only)",
    lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 * 
                 np.exp(-np.std([csrank(g[1][t]).values, csrank(g[2][t]).values, csrank(g[3][t]).values, csrank(g[4][t]).values], axis=0))
))

# Strategy D: Feature 1 & 3 focus with F6 relative spread
for w_spread in [0.1, 0.2, 0.3, 0.4]:
    candidates.append((
        f"F1+F3 Core + {w_spread}*(F6-F1)",
        lambda g, t, ws=w_spread: -(csrank(g[1][t]) + csrank(g[3][t])) / 2.0 + ws * csrank(g[6][t] - g[1][t])
    ))

# Strategy E: Cross-sectional z-score tails
candidates.append((
    "Tanh Z-score Core (scale=1.5)",
    lambda g, t: np.tanh(-1.5 * (zscore(g[1][t]) + zscore(g[2][t]) + zscore(g[3][t]) + zscore(g[4][t])) / 4.0)
))

# Strategy F: Reversion + Feature Volatility normalizer
candidates.append((
    "Vol-Normalized Reversion (rolling 20 vol)",
    lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 / 
                 (csrank(g[1][t]).diff(1).abs().rolling(15, min_periods=1).mean() + 0.1)
))

print("=" * 110)
print(f"{'Alpha Formulation':<45} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Min Dist':<9} | {'Nearest Signal':<20}")
print("=" * 110)

best_proposals = []
for name, func in candidates:
    all_ic = []
    all_pnl = []
    all_gf = []

    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tgt = item["val_target"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}

        raw = func(g, tickers)
        if isinstance(raw, np.ndarray):
            raw = pd.DataFrame(raw, index=feat.index, columns=tickers)
        sig = raw.sub(raw.mean(axis=1), axis=0)

        # IC
        dot = (sig * tgt[tickers]).sum(axis=1)
        norm_s = np.sqrt((sig**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        # Sharpe
        smoothed = sig.ewm(span=200, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret[tickers])
        all_pnl.append(pnl)

        # Gauge
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
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

    print(f"{name:<45} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | {min_dist_deg:>7.1f}° | {nearest_id:<20}")
    if net_sh > 0.015:
        best_proposals.append((name, net_sh, np.mean(all_ic), min_dist_deg, nearest_id))
