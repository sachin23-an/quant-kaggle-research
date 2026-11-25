from data_loader import list_periods, load_period
from evaluation import sharpe_ratio
import pandas as pd
import numpy as np
import xgboost as xgb

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def compute_ic(pred, target):
    p = pred.sub(pred.mean(axis=1), axis=0)
    t = target.sub(target.mean(axis=1), axis=0)
    dot = (p * t).sum(axis=1)
    norm_p = np.sqrt((p ** 2).sum(axis=1)) + 1e-8
    norm_t = np.sqrt((t ** 2).sum(axis=1)) + 1e-8
    return (dot / (norm_p * norm_t)).mean()

# Test across periods 022 to 031 (last 10 periods)
periods = list_periods('data/train', 'small')[-10:]
print(f"Testing Spread-Aware XGBoost across recent {len(periods)} periods...", flush=True)

val_ics = []
val_sharpes = []

for p in periods:
    d = load_period(p, 'data/train', 'small')
    feat = d['features']
    targ = d['target']
    rets = d['returns']
    tickers = list(targ.columns)
    
    n_val = min(800, int(0.2 * len(rets)))
    # Subsample training to last 10,000 rows for speed and recency
    tr_feat = feat.iloc[-n_val-10000:-n_val]
    tr_targ = targ.iloc[-n_val-10000:-n_val]
    va_feat = feat.iloc[-n_val:]
    va_targ = targ.iloc[-n_val:]
    va_rets = rets.iloc[-n_val:]
    
    def get_features(df):
        g = {i: df[f'Feature.{i}'][tickers] for i in range(1, 7)}
        b = {}
        # Core signals
        b['f6'] = csrank(g[6])
        b['f5'] = csrank(g[5])
        b['f_core'] = (csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
        
        # Primary alpha spreads against F6
        b['f5_f6'] = csrank(g[5] - g[6])
        b['f4_f6'] = csrank(g[4] - g[6])
        b['f2_f6'] = csrank(g[2] - g[6])
        b['f1_f6'] = csrank(g[1] - g[6])
        b['f3_f6'] = csrank(g[3] - g[6])
        
        X = np.hstack([v.values.reshape(-1, 1) for v in b.values()])
        return X
        
    X_tr = get_features(tr_feat)
    y_tr = tr_targ[tickers].values.reshape(-1)
    ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 2,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 500,
        'reg_lambda': 5.0,
        'verbosity': 0,
        'nthread': 4,
    }
    dtrain = xgb.DMatrix(X_tr[ok_tr], label=y_tr[ok_tr])
    bst = xgb.train(params, dtrain, num_boost_round=30)
    
    X_va = get_features(va_feat)
    raw_preds = bst.predict(xgb.DMatrix(np.nan_to_num(X_va)))
    pred_df = pd.DataFrame(raw_preds.reshape(-1, len(tickers)), index=va_feat.index, columns=tickers)
    
    # Calculate raw IC
    ic = compute_ic(pred_df, va_targ[tickers])
    val_ics.append(ic)
    
    # Position smoothing (span=120) and causal tilt removal
    sig = pred_df.ewm(span=120, min_periods=1).mean()
    exp_m = sig.shift(1).expanding(min_periods=1).mean()
    sig = (sig - exp_m).fillna(0.0)
    sig = sig.sub(sig.mean(axis=1), axis=0)
    
    sh = sharpe_ratio(sig, va_rets[tickers])
    val_sharpes.append(sh)
    print(f"Period {p}: IC={ic:+.4f}, Sharpe={sh:+.4f}", flush=True)

print(f"\n--- Validation Results Across Recent 10 Periods ---")
print(f"Mean Validation IC:     {np.mean(val_ics):+.4f} (min={np.min(val_ics):+.4f}, max={np.max(val_ics):+.4f})")
print(f"Mean Validation Sharpe: {np.mean(val_sharpes):+.4f} (Positive: {np.mean([s>0 for s in val_sharpes])*100:.1f}%)")
