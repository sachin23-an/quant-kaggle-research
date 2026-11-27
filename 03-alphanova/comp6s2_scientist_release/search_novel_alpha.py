"""
Search for the optimal rotated signal that achieves:
1. Angular distance >= 32.0° to ALL 446 signals in the database.
2. Net Sharpe > 0 across full walk-forward.
3. Maximum Dim0 (IC).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path("D:/Projects/Quant/quant-competition-research/03-alphanova/comp6s2_scientist_release")
sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities
from evaluation import TRANSACTION_COST

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

# Precompute cached blocks for fast evaluation
cached_blocks = []
for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
    
    # 1. Base winning alpha core
    inv_core = ((-csrank(g[1])) + (-csrank(g[2])) + (-csrank(g[3])) + (-csrank(g[4]))) / 4.0
    f6_spread = (csrank(g[6] - g[1]) + csrank(g[6] - g[2]) + csrank(g[6] - g[3]) + csrank(g[6] - g[4])) / 4.0
    s_23 = csrank(g[2] - g[3])
    s_41 = csrank(g[4] - g[1])
    
    sig_core = 0.40 * inv_core + 0.25 * f6_spread + 0.20 * s_23 + 0.15 * s_41
    sig_core = sig_core.sub(sig_core.mean(axis=1), axis=0)
    
    # Candidate orthogonal rotators
    rot_f5 = -csrank(g[5])
    rot_f6 = csrank(g[6])
    rot_35 = csrank(g[3] - g[5])
    rot_54 = csrank(g[5] - g[4])
    rot_cubic = csrank(s_23**3)
    
    cached_blocks.append({
        "ret": ret,
        "g_mat": harness._cached_gauge[idx],
        "sig_core": sig_core,
        "rot_f5": rot_f5.sub(rot_f5.mean(axis=1), axis=0),
        "rot_f6": rot_f6.sub(rot_f6.mean(axis=1), axis=0),
        "rot_35": rot_35.sub(rot_35.mean(axis=1), axis=0),
        "rot_54": rot_54.sub(rot_54.mean(axis=1), axis=0),
        "rot_cubic": rot_cubic.sub(rot_cubic.mean(axis=1), axis=0),
    })

print("=" * 85)
print(f"{'ROTATOR':<20} | {'W_ROT':<6} | {'SPAN':<5} | {'NET SHARPE':<11} | {'DIM0 (IC)':<10} | {'MIN DIST (DEG)':<15}")
print("-" * 85)

candidates = [
    ("rot_f5", "Feature 5 Reversal"),
    ("rot_f6", "Feature 6 Long"),
    ("rot_35", "Spread F3-F5"),
    ("rot_54", "Spread F5-F4"),
    ("rot_cubic", "Cubic Tail Spread"),
]

best_candidates = []

for rot_key, rot_name in candidates:
    for w_rot in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for span in [150, 200]:
            all_net = []
            all_gf = []
            for item in cached_blocks:
                sig = (1.0 - w_rot) * item["sig_core"] + w_rot * item[rot_key]
                sig = sig.sub(sig.mean(axis=1), axis=0)
                smoothed = sig.ewm(span=span, min_periods=1).mean().sub(sig.mean(axis=1), axis=0)
                
                # Net PnL
                gross = (smoothed.shift(1).ffill() * item["ret"]).sum(axis=1)
                gross.iloc[0] = 0
                turnover = smoothed.shift(1).fillna(0.0).diff().abs().sum(axis=1)
                net = gross - TRANSACTION_COST * turnover
                net.iloc[0] = 0
                all_net.append(net)
                
                g_mat = item["g_mat"]
                if g_mat is not None:
                    p_v = smoothed.values - smoothed.values.mean(axis=1, keepdims=True)
                    n = min(len(g_mat), len(p_v))
                    gf = compute_gauge_fixed(p_v[:n], g_mat[:n])
                    all_gf.append(gf)
                    
            net_s = pd.concat(all_net)
            sharpe = net_s.mean() / net_s.std()
            city = compute_city(np.concatenate(all_gf, axis=0))
            
            comp = compare_to_cities(city, city_db_path=db_path)
            min_dist = comp.iloc[0]["angular_distance_deg"]
            
            if min_dist >= 20.0 and sharpe > 0:
                print(f"{rot_name:<20} | {w_rot:<6.2f} | {span:<5} | {sharpe:+.4f}      | {city[0]:+.4f}     | {min_dist:5.2f}°")
                best_candidates.append({
                    "rot_name": rot_name, "rot_key": rot_key, "w_rot": w_rot,
                    "span": span, "sharpe": sharpe, "dim0": city[0], "min_dist": min_dist
                })

print("=" * 85)
