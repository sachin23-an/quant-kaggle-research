# Quantitative Trading & Alpha Research Benchmarks

Personal quantitative research benchmarks on financial datasets. Focused on order-book microstructure, volatility forecasting, and systematic alpha modeling.

Developed by **Sachin Kumar** ([GitHub: sachin23-an](https://github.com/sachin23-an)).

---

## Projects Overview

| Project | Domain | Metric | Result |
| :--- | :--- | :--- | :---: |
| **01. Optiver - Realized Volatility Prediction** | Order Book Realized Volatility | RMSPE | *In Progress* |
| **02. Optiver - Trading at the Close** | Auction Imbalance & Microstructure | MAE | **5.501 bps** (vs 5.952 Baseline) |

---

## 1. Optiver: Realized Volatility Prediction
* **Problem:** Predict 10-minute forward realized volatility for hundreds of stocks from book and trade data.
* **Key Features:** Realized volatility of WAP, bid-ask spreads, and trade volume.
* **Evaluation:** Evaluated by Kaggle using RMSPE via clean `submission.csv`.

---

## 2. Optiver: Trading at the Close
* **Problem:** Predict stock price movements during the closing auction relative to the market index.
* **Key Features:** Order book depth imbalance, Stoikov micro-price, and cross-sectional market-neutralized WAP.
* **Result:** Achieved **5.501 bps** validation MAE (vs. global winning benchmark of 5.403 bps).

---

## Tech Stack
* **Python:** Pandas, NumPy, Scikit-learn, LightGBM
* **Platform:** Kaggle Notebooks
