from data_loader import list_periods, load_period
from evaluation import sharpe_ratio
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def xs_demean(df):
    return df.sub(df.mean(axis=1), axis=0)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

periods = list_periods('data/train', 'small')

# Test across all 31 periods:
# We want to find: what feature combination or model achieves POSITIVE walkforward Sharpe across the validation sets?

# Let's test individual features without expanding mean vs with expanding mean, varying span
results = {}

for span in [48, 96, 144, 240, 360]:
    for f in range(1, 7):
        sh_pos = []
        sh_neg = []
        for p in periods:
            data = load_period(p, 'data/train', 'small')
            n_val = min(800, int(0.2 * len(data['returns'])))
            tickers = list(data['returns'].columns)
            val_f = data['features'][f'Feature.{f}'][tickers].iloc[-n_val:]
            val_r = data['returns'][tickers].iloc[-n_val:]
            
            rk = csrank(val_f)
            s = xs_demean(rk.ewm(span=span).mean())
            
            # positive
            sh_p = sharpe_ratio(s, val_r)
            sh_n = sharpe_ratio(-s, val_r)
            sh_pos.append(sh_p)
            sh_neg.append(sh_n)
            
        results[(span, f, '+')] = np.mean(sh_pos)
        results[(span, f, '-')] = np.mean(sh_neg)

sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 15 single-feature validation Sharpes:")
for (span, f, sign), sh in sorted_res[:15]:
    print(f"Span {span:3d}, {sign}Feature.{f}: Mean Sharpe = {sh:+.4f}")

print("\nBottom 10 single-feature validation Sharpes:")
for (span, f, sign), sh in sorted_res[-10:]:
    print(f"Span {span:3d}, {sign}Feature.{f}: Mean Sharpe = {sh:+.4f}")
