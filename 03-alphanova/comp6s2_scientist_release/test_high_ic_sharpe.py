from data_loader import list_periods, load_period
from evaluation import sharpe_ratio
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge

def xs_demean(df):
    return df.sub(df.mean(axis=1), axis=0)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def compute_ic(pred, target):
    p = xs_demean(pred)
    t = xs_demean(target)
    # Cosine similarity per row
    dot = (p * t).sum(axis=1)
    norm_p = np.sqrt((p ** 2).sum(axis=1)) + 1e-8
    norm_t = np.sqrt((t ** 2).sum(axis=1)) + 1e-8
    return (dot / (norm_p * norm_t)).mean()

# Test on Period 031
data = load_period('031', 'data/train', 'small')
feat = data['features']
target = data['target']
returns = data['returns']
tickers = list(returns.columns)

# Train on earlier rows, test on last 800 rows
n_val = 800
train_feat = {i: feat[f'Feature.{i}'][tickers].iloc[:-n_val] for i in range(1, 7)}
train_target = target[tickers].iloc[:-n_val]

val_feat = {i: feat[f'Feature.{i}'][tickers].iloc[-n_val:] for i in range(1, 7)}
val_target = target[tickers].iloc[-n_val:]
val_returns = returns[tickers].iloc[-n_val:]

print("Testing feature combinations against target and returns on Period 031 holdout...")

# Let's inspect pairwise correlation with target across all features
for i in range(1, 7):
    r_val = csrank(val_feat[i])
    ic = compute_ic(r_val, val_target)
    sig = r_val.ewm(span=240).mean()
    sig = xs_demean(sig)
    sh = sharpe_ratio(sig, val_returns)
    print(f"Feature.{i}: IC(target)={ic:+.4f}, Sharpe(returns)={sh:+.4f}")

# Cross-feature spreads
for i in range(1, 7):
    for j in range(i+1, 7):
        diff = csrank(val_feat[i]) - csrank(val_feat[j])
        ic = compute_ic(diff, val_target)
        sig = xs_demean(diff.ewm(span=240).mean())
        sh = sharpe_ratio(sig, val_returns)
        if abs(ic) > 0.03 or abs(sh) > 0.02:
            print(f"F{i} - F{j}: IC={ic:+.4f}, Sharpe={sh:+.4f}")
