"""
Fast parameter & feature optimizer across all 31 cached periods.
Evaluates in pure vectorized numpy (takes < 3 seconds).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path("D:/Projects/Quant/quant-competition-research/03-alphanova/comp6s2_scientist_release")
sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import TRANSACTION_COST
from city_tools import compute_gauge_fixed, compute_city

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def get_block_dict(feat, tickers):
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
    b = {}
    # Top pairwise spreads
    b["f2_f3"] = csrank(g[2] - g[3])
    b["f4_f1"] = csrank(g[4] - g[1])
    b["f2_f1"] = csrank(g[2] - g[1])
    b["f4_f3"] = csrank(g[4] - g[3])
    b["f6_f4"] = csrank(g[6] - g[4])
    b["f6_f1"] = csrank(g[6] - g[1])
    b["f6_f2"] = csrank(g[6] - g[2])
    b["f6_f3"] = csrank(g[6] - g[3])
    b["inv_f1"] = -csrank(g[1])
    b["inv_f2"] = -csrank(g[2])
    b["inv_f3"] = -csrank(g[3])
    b["inv_f4"] = -csrank(g[4])
    return b

# Test pure static linear blend first vs Ridge
print("=" * 80)
print(f"{'STRATEGY':<35} | {'SPAN':<5} | {'NET SHARPE':<11} | {'GROSS SHARPE':<12} | {'DIM0 (IC)':<10}")
print("-" * 80)

for span in [100, 150, 200, 250, 300, 400, 500]:
    # Test blend of top 4 spreads:
    # 0.35 * f2_f3 + 0.25 * f4_f1 + 0.25 * f2_f1 + 0.15 * f4_f3
    all_net = []
    all_gross = []
    all_gf = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        b = get_block_dict(feat, tickers)
        
        # Test weighted combination
        sig = 0.35 * b["f2_f3"] + 0.25 * b["f4_f1"] + 0.25 * b["f2_f1"] + 0.15 * b["f4_f3"]
        sig = sig.sub(sig.mean(axis=1), axis=0)
        smoothed = sig.ewm(span=span, min_periods=1).mean().sub(sig.mean(axis=1), axis=0)
        
        # Net & Gross PnL
        gross = (smoothed.shift(1).ffill() * ret).sum(axis=1)
        gross.iloc[0] = 0
        turnover = smoothed.shift(1).fillna(0.0).diff().abs().sum(axis=1)
        net = gross - TRANSACTION_COST * turnover
        net.iloc[0] = 0
        
        all_net.append(net)
        all_gross.append(gross)
        
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_v = smoothed.values - smoothed.values.mean(axis=1, keepdims=True)
            n = min(len(g_mat), len(p_v))
            gf = compute_gauge_fixed(p_v[:n], g_mat[:n])
            all_gf.append(gf)
            
    net_s = pd.concat(all_net)
    gross_s = pd.concat(all_gross)
    n_sharpe = net_s.mean() / net_s.std()
    g_sharpe = gross_s.mean() / gross_s.std()
    city = compute_city(np.concatenate(all_gf, axis=0))
    print(f"{'Top Spreads Static Blend':<35} | {span:<5} | {n_sharpe:+.4f}      | {g_sharpe:+.4f}       | {city[0]:+.4f}")

# Now test inverse valuation core:
print("-" * 80)
for span in [100, 150, 200, 250, 300, 400, 500]:
    all_net = []
    all_gross = []
    all_gf = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        b = get_block_dict(feat, tickers)
        
        # Inverted core + F6 spreads
        sig = (b["inv_f1"] + b["inv_f2"] + b["inv_f3"] + b["inv_f4"]) / 4.0 + 0.5 * (b["f6_f1"] + b["f6_f2"] + b["f6_f3"] + b["f6_f4"]) / 4.0
        sig = sig.sub(sig.mean(axis=1), axis=0)
        smoothed = sig.ewm(span=span, min_periods=1).mean().sub(sig.mean(axis=1), axis=0)
        
        gross = (smoothed.shift(1).ffill() * ret).sum(axis=1)
        gross.iloc[0] = 0
        turnover = smoothed.shift(1).fillna(0.0).diff().abs().sum(axis=1)
        net = gross - TRANSACTION_COST * turnover
        net.iloc[0] = 0
        
        all_net.append(net)
        all_gross.append(gross)
        
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_v = smoothed.values - smoothed.values.mean(axis=1, keepdims=True)
            n = min(len(g_mat), len(p_v))
            gf = compute_gauge_fixed(p_v[:n], g_mat[:n])
            all_gf.append(gf)
            
    net_s = pd.concat(all_net)
    gross_s = pd.concat(all_gross)
    n_sharpe = net_s.mean() / net_s.std()
    g_sharpe = gross_s.mean() / gross_s.std()
    city = compute_city(np.concatenate(all_gf, axis=0))
    print(f"{'Inverse Core + F6 Spreads':<35} | {span:<5} | {n_sharpe:+.4f}      | {g_sharpe:+.4f}       | {city[0]:+.4f}")
print("=" * 80)
