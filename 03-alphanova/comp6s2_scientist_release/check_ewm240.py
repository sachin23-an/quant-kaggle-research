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
print(f"Testing across all {len(periods)} periods...")

results = {f"F{i}": [] for i in range(1, 7)}
results_inv = {f"-F{i}": [] for i in range(1, 7)}

for p in periods:
    data = load_period(p, 'data/train', 'small')
    features = data['features']
    returns = data['returns']
    tickers = list(returns.columns)
    
    for f_idx in range(1, 7):
        # Raw features with EWM 240
        raw = xs_demean(features[f'Feature.{f_idx}'][tickers])
        s = xs_demean(raw.ewm(span=240).mean())
        g, n = gross_and_net(s, returns[tickers])
        results[f"F{f_idx}"].append(n)
        
        # Inverted
        g_inv, n_inv = gross_and_net(-s, returns[tickers])
        results_inv[f"-F{f_idx}"].append(n_inv)

print("\n--- Net Sharpe across all periods (EWM 240) ---")
for k, v in results.items():
    print(f"{k}: mean net Sharpe = {np.mean(v):+.4f}, positive periods = {np.mean([x>0 for x in v])*100:.1f}%")

print("\n--- Inverted Net Sharpe across all periods (EWM 240) ---")
for k, v in results_inv.items():
    print(f"{k}: mean net Sharpe = {np.mean(v):+.4f}, positive periods = {np.mean([x>0 for x in v])*100:.1f}%")
