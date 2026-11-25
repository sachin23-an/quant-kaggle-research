import numpy as np
import pandas as pd
from data_loader import list_periods, load_period
from evaluation import sharpe_ratio

DATA_DIR = "data/train"
UNIVERSE = "small"

periods = list_periods(DATA_DIR, UNIVERSE)

sharpes = []

for p in periods:
    data = load_period(p, DATA_DIR, UNIVERSE)
    feat = data["features"]
    returns = data["returns"]
    tickers = list(returns.columns)
    
    # Pure positive alpha subspace
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
    
    # Cross-sectional ranks
    r = {}
    for i in range(1, 7):
        rk = g[i].rank(axis=1)
        r[i] = (rk - rk.mean(axis=1).values.reshape(-1, 1)) / (rk.std(axis=1).values.reshape(-1, 1) + 1e-8)
        
    # Top verified positive signals
    s1 = -r[6]
    s2 = (g[5] - g[6]).rank(axis=1)
    s2 = (s2 - s2.mean(axis=1).values.reshape(-1, 1)) / (s2.std(axis=1).values.reshape(-1, 1) + 1e-8)
    
    s3 = (g[4] - g[6]).rank(axis=1)
    s3 = (s3 - s3.mean(axis=1).values.reshape(-1, 1)) / (s3.std(axis=1).values.reshape(-1, 1) + 1e-8)
    
    s4 = (g[5] - g[1]).rank(axis=1)
    s4 = (s4 - s4.mean(axis=1).values.reshape(-1, 1)) / (s4.std(axis=1).values.reshape(-1, 1) + 1e-8)
    
    # Equal-weighted pure alpha composite
    composite = 0.40 * s1 + 0.30 * s2 + 0.15 * s3 + 0.15 * s4
    
    # Optimal turnover smoothing (span=120)
    sig = composite.ewm(span=120, min_periods=1).mean()
    sig = sig.sub(sig.mean(axis=1), axis=0)
    
    sh = sharpe_ratio(sig, returns[tickers])
    sharpes.append(sh)

print(f"PURE ALPHA COMPOSITE MEAN SHARPE (across all 31 periods): {np.mean(sharpes):+.4f}")
print(f"Percentage of profitable periods: {np.mean([s > 0 for s in sharpes])*100:.1f}%")
print(f"Sharpes: min={np.min(sharpes):+.4f}, max={np.max(sharpes):+.4f}")
