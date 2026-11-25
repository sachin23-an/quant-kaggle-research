from data_loader import list_periods, load_period
import pandas as pd
import numpy as np

def xs_demean(df):
    return df.sub(df.mean(axis=1), axis=0)

def gross_and_net(signal, rets):
    gross = (signal.shift(1) * rets).sum(axis=1)
    turnover = signal.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    cost = 5e-4 * turnover
    net = gross - cost
    return gross.mean() / (gross.std() + 1e-8), net.mean() / (net.std() + 1e-8)

periods = list_periods('data/train', 'small')
print(f"Loading {len(periods)} periods into memory...", flush=True)

loaded_data = []
for p in periods:
    data = load_period(p, 'data/train', 'small')
    features = data['features']
    returns = data['returns']
    tickers = list(returns.columns)
    
    f_core = xs_demean(
        xs_demean(features['Feature.1'][tickers]) +
        xs_demean(features['Feature.2'][tickers]) +
        xs_demean(features['Feature.3'][tickers]) +
        xs_demean(features['Feature.4'][tickers])
    ) / 4.0
    
    f_inv6 = -xs_demean(features['Feature.6'][tickers])
    loaded_data.append((f_core, f_inv6, returns[tickers]))

print("Data loaded! Running grid search...", flush=True)

for span in [120, 180, 240, 360]:
    for w_inv6 in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]:
        sharpes = []
        for f_core, f_inv6, rets in loaded_data:
            composite = (1.0 - w_inv6) * f_core + w_inv6 * f_inv6
            sig = xs_demean(composite.ewm(span=span).mean())
            _, n = gross_and_net(sig, rets)
            sharpes.append(n)
            
        print(f"Span={span}, w_inv6={w_inv6:.1f}: Mean Net Sharpe = {np.mean(sharpes):+.4f}, Pos% = {np.mean([s>0 for s in sharpes])*100:.1f}%, min={np.min(sharpes):+.4f}, max={np.max(sharpes):+.4f}", flush=True)
