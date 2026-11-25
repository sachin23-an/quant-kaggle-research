from data_loader import list_periods, load_period
from evaluation import sharpe_ratio
import pandas as pd
import numpy as np

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

periods = list_periods('data/train', 'small')[-10:]

# Test thresholding / conviction gating:
# If cross-sectional feature dispersion is below threshold, scale down signal to zero (avoid choppy turnover)

for q in [0.0, 0.2, 0.3, 0.4, 0.5]:
    sh_list = []
    for p in periods:
        d = load_period(p, 'data/train', 'small')
        rets = d['returns']
        tickers = list(rets.columns)
        n_val = min(800, int(0.2 * len(rets)))
        va_feat = d['features'].iloc[-n_val:]
        va_rets = rets.iloc[-n_val:]
        
        # Cross-sectional spread signal
        s = csrank(va_feat['Feature.5'][tickers] - va_feat['Feature.6'][tickers])
        
        # Compute contemporaneous cross-sectional dispersion of Feature 6
        disp = va_feat['Feature.6'][tickers].std(axis=1)
        thresh = disp.quantile(q)
        
        # Gate: only trade when dispersion > thresh
        gate = (disp > thresh).astype(float).values.reshape(-1, 1)
        gated_s = s * gate
        
        sig = gated_s.ewm(span=120, min_periods=1).mean()
        sig = sig.sub(sig.mean(axis=1), axis=0)
        
        sh = sharpe_ratio(sig, va_rets[tickers])
        sh_list.append(sh)
        
    print(f"Quantile Thresh={q:.1f}: Mean Sharpe = {np.mean(sh_list):+.4f}, Pos%={np.mean([x>0 for x in sh_list])*100:.0f}%")
