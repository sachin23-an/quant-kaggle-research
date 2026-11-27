"""Find signals with high Dim0 (high IC) and explore their distance to top signals."""

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

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

# Test various combinations of features, non-linearities, ratios, and lags
candidates = {}

# 1. Rank inversions of features
candidates["Inv F1..F4 equal"] = lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0

# 2. Can we use raw z-score instead of rank?
def zscore(df):
    m = df.mean(axis=1).values.reshape(-1, 1)
    s = df.std(axis=1).values.reshape(-1, 1) + 1e-8
    return (df - m) / s

candidates["Zscore -(F1..F4)"] = lambda g, t: -(zscore(g[1][t]) + zscore(g[2][t]) + zscore(g[3][t]) + zscore(g[4][t])) / 4.0

# 3. What if we weight by asset volatility or inverse volatility?
candidates["Zscore -(F1+F3)"] = lambda g, t: -(zscore(g[1][t]) + zscore(g[3][t])) / 2.0

# 4. What if we combine inverted features with momentum of features (delta)?
candidates["Inv Core - 0.2*diff(Core)"] = lambda g, t: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 - 0.2 * (csrank(g[1][t]).diff(1).fillna(0) + csrank(g[3][t]).diff(1).fillna(0)) / 2.0

# 5. What if we use 2-period lag?
candidates["Inv Core (lag 1)"] = lambda g, t: -(csrank(g[1][t].shift(1).fillna(0)) + csrank(g[2][t].shift(1).fillna(0)) + csrank(g[3][t].shift(1).fillna(0)) + csrank(g[4][t].shift(1).fillna(0))) / 4.0

# 6. What about non-linear sigmoid / softmax / tanh of core?
candidates["Tanh Core"] = lambda g, t: np.tanh(-2.0 * (zscore(g[1][t]) + zscore(g[2][t]) + zscore(g[3][t]) + zscore(g[4][t])) / 4.0)

# 7. What about cross-sectional standardization of product?
candidates["Rank Product -(F1*F3)"] = lambda g, t: -csrank(csrank(g[1][t]) * csrank(g[3][t]))

# 8. What about F6 combined with F5?
candidates["F6 - F5"] = lambda g, t: csrank(g[6][t] - g[5][t])

print("=" * 110)
print(f"{'Candidate':<35} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'City Dim0':<10} | {'Min Dist':<9} | {'Nearest ID':<20}")
print("=" * 110)

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

        dot = (sig * tgt[tickers]).sum(axis=1)
        norm_s = np.sqrt((sig**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        all_ic.append(ic)

        smoothed = sig.ewm(span=200, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret[tickers])
        all_pnl.append(pnl)

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

    print(f"{name:<35} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | {city[0]:>+10.4f} | {min_dist_deg:>7.1f}° | {nearest_id:<20}")
