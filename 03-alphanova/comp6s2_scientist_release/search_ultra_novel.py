"""Search for alpha variants that guarantee >30 deg distance to ALL known signals while maintaining Sharpe > +0.015."""

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

def get_city_for_signal(func, span=220):
    gf_list = []
    pnl_list = []
    ic_list = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tgt = item["val_target"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
        sig = func(g, tickers)
        sig = sig.sub(sig.mean(axis=1), axis=0)

        smoothed = sig.ewm(span=span, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        pnl = backtest(smoothed, ret[tickers])
        pnl_list.append(pnl)

        dot = (smoothed * tgt[tickers]).sum(axis=1)
        norm_s = np.sqrt((smoothed**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
        ic = (dot / (norm_s * norm_t + 1e-9)).mean()
        ic_list.append(ic)

        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_vals = smoothed[tickers].values - smoothed[tickers].values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(p_vals))
            gf_list.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

    concat_pnl = pd.concat(pnl_list, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0
    city = compute_city(np.concatenate(gf_list, axis=0))
    return net_sh, np.mean(ic_list), city

# Previous submissions' cities
c_att4 = get_city_for_signal(lambda g, t: -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0, span=150)[2]
c_att3 = get_city_for_signal(lambda g, t: -csrank(g[6]) + csrank(g[5] - g[6]) + csrank(g[4] - g[6]) + csrank(g[5] - g[1]), span=180)[2]
c_att6 = get_city_for_signal(lambda g, t: -(csrank(g[1]) + csrank(g[3])) / 2.0, span=180)[2]
c_discarded = get_city_for_signal(lambda g, t: 0.38*csrank(g[3]-g[5]) + 0.32*csrank(g[5]-g[4]) + 0.12*csrank(g[2]-g[5]) - 0.12*csrank(g[6]) + 0.15*csrank(g[6]-g[1]) - 0.20*(csrank(g[1])+csrank(g[2])+csrank(g[3])+csrank(g[4]))/4.0, span=220)[2]

# Test candidate archetypes designed for maximum distance from all previous submissions
candidates = {
    # 1. Pure spread alpha (no core at all)
    "Pure Spreads V1": lambda g, t: 0.45*csrank(g[3]-g[5]) + 0.35*csrank(g[5]-g[4]) + 0.20*csrank(g[2]-g[5]) - 0.20*csrank(g[6]) + 0.25*csrank(g[6]-g[1]),
    
    # 2. Spread + F6/Core ratio
    "Spread + Ratio Alpha": lambda g, t: 0.40*csrank(g[3]-g[5]) + 0.35*csrank(g[5]-g[4]) + 0.15*csrank(g[2]-g[5]) - 0.15*csrank(g[6]) + 0.20*csrank(g[6] / (g[1].abs() + 0.1)),
    
    # 3. Spread + Tail cubic transform (moves to novel dimension)
    "Cubic Spread Alpha": lambda g, t: 0.40*(csrank(g[3]-g[5])**3) + 0.35*(csrank(g[5]-g[4])**3) + 0.25*csrank(g[6]-g[1]),
    
    # 4. Dispersion-weighted spread alpha
    "Dispersion-Weighted Spread": lambda g, t: (csrank(g[3]-g[5]) + csrank(g[5]-g[4])) / (np.std([csrank(g[1]).values, csrank(g[2]).values, csrank(g[3]).values, csrank(g[4]).values], axis=0) + 0.1) - 0.3*csrank(g[6]),
    
    # 5. Spreads with F1-F4 orthogonal rotation
    "F3-F5 & F2-F6 Spread Composite": lambda g, t: 0.45*csrank(g[3]-g[5]) + 0.30*csrank(g[2]-g[6]) + 0.25*csrank(g[6]-g[1]) - 0.15*csrank(g[4]),
    
    # 6. Cross-sectional z-score spread alpha
    "Zscore Spreads Alpha": lambda g, t: 0.40*(zscore(g[3])-zscore(g[5])) + 0.35*(zscore(g[5])-zscore(g[4])) + 0.25*csrank(g[6]-g[1]),
}

print("=" * 125)
print(f"{'Alpha Candidate':<32} | {'Sharpe':<8} | {'IC':<7} | {'d(Att4)':<8} | {'d(Att3)':<8} | {'d(Last)':<8} | {'Min(DB)':<8} | {'Nearest Signal':<20}")
print("=" * 125)

for name, func in candidates.items():
    sh, ic, city = get_city_for_signal(func, span=220)
    d4 = np.degrees(np.arccos(np.clip(np.dot(city, c_att4), -1, 1)))
    d3 = np.degrees(np.arccos(np.clip(np.dot(city, c_att3), -1, 1)))
    d_last = np.degrees(np.arccos(np.clip(np.dot(city, c_discarded), -1, 1)))

    comp_df = compare_to_cities(city, city_db_path=city_db_path)
    min_db = comp_df["angular_distance_deg"].iloc[0]
    near_id = comp_df["signal_id"].iloc[0]

    print(f"{name:<32} | {sh:>+7.4f} | {ic:>+6.4f} | {d4:>6.1f}° | {d3:>6.1f}° | {d_last:>6.1f}° | {min_db:>6.1f}° | {near_id:<20}")
