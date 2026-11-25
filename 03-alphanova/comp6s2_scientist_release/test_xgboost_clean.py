import numpy as np
import pandas as pd
import xgboost as xgb
from data_loader import list_periods, load_period
from evaluation import sharpe_ratio

DATA_DIR = "data/train"
UNIVERSE = "small"

periods = list_periods(DATA_DIR, UNIVERSE)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

sharpes = []

for p in periods:
    data = load_period(p, DATA_DIR, UNIVERSE)
    feat = data['features']
    target = data['target']
    returns = data['returns']
    tickers = list(returns.columns)
    
    n_val = min(800, int(0.2 * len(returns)))
    train_feat = feat.iloc[:-n_val]
    train_targ = target.iloc[:-n_val]
    val_feat = feat.iloc[-n_val:]
    val_rets = returns.iloc[-n_val:]
    
    # Feature blocks on train
    X_train_blocks = [csrank(train_feat[f'Feature.{i}'][tickers]).values.reshape(-1, 1) for i in range(1, 7)]
    X_tr = np.hstack(X_train_blocks)
    y_tr = train_targ[tickers].values.reshape(-1)
    ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)
    
    # Train low-capacity XGBoost
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 2,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 500,
        'reg_lambda': 10.0,
        'verbosity': 0,
        'nthread': 4,
    }
    dtrain = xgb.DMatrix(X_tr[ok_tr], label=y_tr[ok_tr])
    bst = xgb.train(params, dtrain, num_boost_round=30)
    
    # Predict on val
    X_val_blocks = [csrank(val_feat[f'Feature.{i}'][tickers]).values.reshape(-1, 1) for i in range(1, 7)]
    X_val = np.hstack(X_val_blocks)
    preds = bst.predict(xgb.DMatrix(np.nan_to_num(X_val)))
    pred_df = pd.DataFrame(preds.reshape(-1, len(tickers)), index=val_feat.index, columns=tickers)
    
    # Turnover control
    pred_df = pred_df.ewm(span=120, min_periods=1).mean()
    exp_mean = pred_df.shift(1).expanding(min_periods=1).mean()
    pred_df = (pred_df - exp_mean).fillna(0.0)
    pred_df = pred_df.sub(pred_df.mean(axis=1), axis=0)
    
    sh = sharpe_ratio(pred_df, val_rets[tickers])
    sharpes.append(sh)

print(f"\nClean 6-Feature XGBoost Mean Sharpe across all {len(periods)} periods: {np.mean(sharpes):+.4f}")
print(f"Positive periods: {np.mean([s > 0 for s in sharpes])*100:.1f}%")
print(f"Latest period (031) Sharpe: {sharpes[-1]:+.4f}")
