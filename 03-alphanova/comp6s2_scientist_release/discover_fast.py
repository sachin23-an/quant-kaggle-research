from data_loader import list_periods, load_period
from evaluation import sharpe_ratio
import pandas as pd
import numpy as np

def xs_demean(df):
    return df.sub(df.mean(axis=1), axis=0)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

periods = list_periods('data/train', 'small')
print(f"Loading {len(periods)} periods...", flush=True)

data_cache = []
for p in periods:
    d = load_period(p, 'data/train', 'small')
    n_val = min(800, int(0.2 * len(d['returns'])))
    tickers = list(d['returns'].columns)
    
    val_f = {i: csrank(d['features'][f'Feature.{i}'][tickers].iloc[-n_val:]) for i in range(1, 7)}
    val_r = d['returns'][tickers].iloc[-n_val:]
    val_t = d['target'][tickers].iloc[-n_val:]
    data_cache.append((val_f, val_r, val_t))

print("Data cached! Testing single features...", flush=True)

for f in range(1, 7):
    for span in [48, 120, 240]:
        sh_p = []
        sh_n = []
        for val_f, val_r, val_t in data_cache:
            s = xs_demean(val_f[f].ewm(span=span).mean())
            sh_p.append(sharpe_ratio(s, val_r))
            sh_n.append(sharpe_ratio(-s, val_r))
        print(f"Feature.{f} (span={span}): +Sharpe={np.mean(sh_p):+.4f} (pos={np.mean([x>0 for x in sh_p])*100:.0f}%), -Sharpe={np.mean(sh_n):+.4f} (pos={np.mean([x>0 for x in sh_n])*100:.0f}%)", flush=True)
