import os
import numpy as np
import pandas as pd
from data_loader import list_periods, load_period
from evaluation import sharpe_ratio

DATA_DIR = "data/train"
UNIVERSE = "small"

periods = list_periods(DATA_DIR, UNIVERSE)

# Test all features 1..6, inverted and non-inverted, smoothed span=120
features_to_test = {}
for i in range(1, 7):
    features_to_test[f"F{i}"] = lambda feat, t, i=i: feat[f"Feature.{i}"][t]
    features_to_test[f"-F{i}"] = lambda feat, t, i=i: -feat[f"Feature.{i}"][t]

for i in range(1, 7):
    for j in range(i + 1, 7):
        features_to_test[f"F{i}-F{j}"] = lambda feat, t, i=i, j=j: feat[f"Feature.{i}"][t] - feat[f"Feature.{j}"][t]
        features_to_test[f"-(F{i}-F{j})"] = lambda feat, t, i=i, j=j: -(feat[f"Feature.{i}"][t] - feat[f"Feature.{j}"][t])

results = {k: [] for k in features_to_test}

for p in periods:
    data = load_period(p, DATA_DIR, UNIVERSE)
    feat = data["features"]
    returns = data["returns"]
    tickers = list(returns.columns)
    
    for name, fn in features_to_test.items():
        val = fn(feat, tickers)
        r = val.rank(axis=1)
        r = (r - r.mean(axis=1).values.reshape(-1, 1)) / (r.std(axis=1).values.reshape(-1, 1) + 1e-8)
        sig = r.ewm(span=120, min_periods=1).mean()
        sig = sig.sub(sig.mean(axis=1), axis=0)
        sh = sharpe_ratio(sig, returns[tickers])
        results[name].append(sh)

mean_sharpes = {k: np.mean(v) for k, v in results.items()}
sorted_sh = sorted(mean_sharpes.items(), key=lambda x: x[1], reverse=True)

print("TOP 15 POSITIVE SHARPE SIGNALS (across all 31 periods):")
for name, sh in sorted_sh[:15]:
    print(f"  {name:15s}: Sharpe = {sh:+.4f}")

print("\nBOTTOM 10 WORST NEGATIVE SHARPE SIGNALS:")
for name, sh in sorted_sh[-10:]:
    print(f"  {name:15s}: Sharpe = {sh:+.4f}")
