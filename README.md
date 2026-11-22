# Systematic Alpha Benchmarks: Quant Research & High-Frequency Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Validation: Purged Walk-Forward](https://img.shields.io/badge/Validation-Purged%20Walk--Forward-orange.svg)]()
[![Look-Ahead Audit: 5-Check Clean](https://img.shields.io/badge/Look--Ahead%20Audit-Passed%20(5%2F5)-brightgreen.svg)]()

A unified quantitative research repository benchmarking systematic alpha forecasting, order-book microstructure signals, and volatility-constrained portfolio allocation across gold-standard financial datasets. 

Developed by **Sachin Kumar** ([GitHub: sachin23-an](https://github.com/sachin23-an)).

---

## 1. Portfolio Architecture & Quant CV Alignment

Recruiters and quant researchers at proprietary trading firms (Jane Street, Citadel, Optiver, Two Sigma, D.E. Shaw) evaluate research candidates on:
1. **Signal extraction on noisy, non-stationary time series**
2. **Defensive validation methodology (zero look-ahead bias, combinatorial/purged splits)**
3. **Execution realism (microstructure, latency markouts, queue fills, transaction-cost sensitivity)**
4. **Risk-adjusted optimization under portfolio constraints**

| Benchmark / Competition | Core Financial Domain | Evaluation Metric | Quant CV Competency Demonstrated |
| :--- | :--- | :--- | :--- |
| **[01-Optiver Closing Auction](./01-optiver-closing-auction/)** | Equity Closing Cross Microstructure | Mean Absolute Error (MAE) & Rank IC | Order-book depth imbalance, micro-price divergence, cross-sectional market neutralization across 200 stocks. |
| **02-Jane Street Market Data** *(In Progress)* | Intraday Cross-Sectional Alpha | Weighted Zero-Mean $R^2$ | Non-stationary responder forecasting, purged group CV, online incremental updating vs static batch models. |
| **03-Hull Tactical Allocation** *(Planned)* | Volatility-Constrained Allocation | Volatility-Penalized Sharpe | Translating return forecasts into dynamic leverage ($0$ to $2\times$) under a strict $120\%$ annualized volatility cap. |

---

## 2. Shared Quantitative Core (`shared/`)

A reusable, production-grade research library enforcing leak-free financial engineering across all benchmarks:

```
shared/
├── validation.py         # PurgedGroupTimeSeriesSplit (De Prado framework with purge & embargo)
├── lookahead_audit.py    # Automated 5-check leakage suite (timestamps, groups, correlations, scalers)
└── metrics.py            # Financial loss functions: Weighted Zero-Mean R², MAE, Rank IC, Sharpe, MDD
```

### Key Engineering Guarantees
* **Purged & Embargoed Cross-Validation**: Prevents serial correlation leakage across forward-looking return windows and eliminates same-day cross-sectional contamination.
* **Automated 5-Check Look-Ahead Audit**:
  1. *Temporal Monotonicity*: Verifies $\max(t_{\text{train}}) + \text{purge} \le \min(t_{\text{test}})$.
  2. *Group Isolation*: Asserts $\text{Train Groups} \cap \text{Test Groups} = \emptyset$.
  3. *Spurious Target Correlation Detector*: Flags any feature with $|r| > 0.85$ to detect future information leakage.
  4. *Rolling Shift Invariance*: Asserts temporal features use strictly causal $t \le T$ inputs.
  5. *Scaler Isolation*: Guarantees preprocessing parameters are fitted strictly on training distributions.

---

## 3. Directory Structure

```
quant-kaggle-research/
├── README.md                              # Master overview and benchmark results
├── requirements.txt                       # Python dependencies (polars, lightgbm, numpy, scipy)
├── .gitignore                             # Git hygiene (ignoring datasets, caches, checkpoints)
├── shared/                                # Reusable quantitative framework
│   ├── __init__.py
│   ├── validation.py                      # Purged & Embargoed Group Time-Series Split
│   ├── lookahead_audit.py                 # Automated Look-Ahead Auditor
│   └── metrics.py                         # Quant metrics (Zero-Mean R², Rank IC, Sharpe)
└── 01-optiver-closing-auction/            # Benchmark 01: Optiver - Trading at the Close
    ├── README.md                          # Methodology, feature ablation table, failure analysis
    ├── notebooks/                         # Exploratory data analysis & feature studies
    └── src/
        ├── features.py                    # Microstructure & market-neutral feature extraction
        └── baseline.py                    # End-to-end leak-free cross-validation pipeline
```

---

## 4. Quickstart & Verification

```bash
# Clone the repository
git clone https://github.com/sachin23-an/quant-kaggle-research.git
cd quant-kaggle-research

# Run shared validation unit tests
python shared/validation.py

# Run automated look-ahead audit test suite
python shared/lookahead_audit.py

# Run shared metrics verification
python shared/metrics.py

# Execute Optiver baseline benchmark pipeline
python 01-optiver-closing-auction/src/baseline.py
```

---

## 5. Contact & Portfolio Links
* **Author**: Sachin Kumar
* **Email**: sachinkumar2310235@gmail.com | 23f1000345@ds.study.iitm.ac.in
* **LinkedIn**: [linkedin.com/in/sachin-kumar-6b5609257](https://www.linkedin.com/in/sachin-kumar-6b5609257/)
* **GitHub Profile**: [github.com/sachin23-an](https://github.com/sachin23-an)
* **Related Projects**:
  * [Latency-to-Execution: Order-Flow Alpha vs. Latency](https://github.com/sachin23-an/latency-to-execution)
  * [The Complexity Trap: Cost Erosion in Systematic Trading](https://github.com/sachin23-an/the-complexity-trap)
  * [Statistical Pairs Trading: Cointegration & Mean Reversion](https://github.com/sachin23-an/pairs-trading-cointegration)
  * [Multi-Method VaR & Portfolio Stress Testing](https://github.com/sachin23-an/multi-method-var-stress-testing)
