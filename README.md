# Quantitative Machine Learning & Alpha Research Benchmarks

Personal quantitative research repository focused on systematic alpha generation, factor neutrality, and machine learning architectures for low signal-to-noise financial data.

Developed by **Sachin Kumar** ([GitHub: @sachin23-an](https://github.com/sachin23-an))  
*BS in Data Science, Indian Institute of Technology Madras ('27)*  
*Former Quantitative Researcher Intern at TLK Industries (Aug–Nov 2026)*

---

## Benchmark Directory

| # | Project | Domain / Platform | Primary Metric | Core Innovation / Result |
| :-: | :--- | :--- | :---: | :--- |
| **01** | [**01-numerai**](./01-numerai) | **Numerai Classic** Live Hedge Fund | CORR20 / MMC20 | Live alpha submission on `v6.0` dataset. Model: [`sachin23_an`](https://numer.ai/sachin23_an). Verified Round 1404 submission. |
| **02** | [**02-credit-card-fraud**](./02-credit-card-fraud) | **Kaggle S6E11** Transaction Fraud | ROC-AUC | Multi-Model Logit-Space Rank Fusion, Clean-Zone Filtering & Constrained Diversity Search. Score: **0.95567** (Rank 81). |

---

## 1. [01. Numerai Classic Hedge Fund Alpha Engine](./01-numerai)

* **Objective:** Predict 20-day forward market-neutral residual returns across 7,000+ global equities for capital allocation by the Numerai hedge fund.
* **The Alpha Challenge:** Extreme low signal-to-noise ratio ($SNR \ll 1$) where raw equity trajectories contain disguised market beta. Success requires orthogonal alpha that is uncorrelated to ~40 known systematic risk factors.
* **Architecture:**
  * **Dataset:** v6.0 ("Unity") architecture dynamically resolved via `NumerAPI`.
  * **Feature Selection:** 101-feature `'small'` orthogonal subset to eliminate collinearity and reduce training memory footprint to $< 1.8$ GB RAM.
  * **Modeling:** Tree-depth constrained `LGBMRegressor` (`max_depth=5`, `num_leaves=31`, `colsample_bytree=0.8`).
  * **Normalization:** Quantile rank transformation mapping signals to uniform $[0, 1]$.
  * **Operational Status:** Live production script with automated scheduling (13:30 UTC / 7:00 PM IST daily release). Verified live submission on model [`sachin23_an`](https://numer.ai/sachin23_an).

👉 **[Read 01-numerai Documentation & Code](./01-numerai)**

---

## 2. [02. Credit Card Fraud Detection](./02-credit-card-fraud)

* **Objective:** Detect fraudulent credit card transactions in synthetic transactional data with severe class imbalance (~1.67% fraud rate).
* **The Methodology & Architecture:**
  * **Logit-Rank Projection:** Converts model outputs into unbounded log-odds space $L(x) = \text{logit}(R(x))$ to combine diverse architectures (RealMLP, TabM, LightGBM, CatBoost, GPT-2) linearly without boundary compression.
  * **Domain Clean-Zone Masking:** Identifies low-risk non-fraud records (clean verification flags, zero disputes, low merchant risk) and shifts their rank downwards, eliminating false-positive contamination.
  * **Constrained Diversity Search:** Searches blend partners under bounded percentile shift ($\le 0.001$) and minimum row reordering ($\ge 10\%$) to find non-duplicative ensemble partners.
* **Leaderboard Result:** Achieved **0.95567 ROC-AUC (Rank 81)** on Kaggle Playground S6E11 across 6 iterations.

👉 **[Read 02-credit-card-fraud Documentation & Code](./02-credit-card-fraud)**

---

## Research Philosophy & Core Principles

1. **Parsimony Over Brute Force:** In noisy and synthetic data regimes, high-capacity models easily overfit noise. Constraining model capacity (`max_depth=5`, orthogonal feature subsets) consistently outperforms deep, unregularized architectures.
2. **Inductive Bias Alignment:** Choosing model representations whose mathematical structure matches the underlying data-generating process (e.g. continuous B-splines for smooth synthetic manifolds) yields superior generalization over brute-force hyperparameter search.
3. **Distribution-Invariant Ensembling:** Blending raw probability estimates from heterogeneous model families causes calibration distortion. Transforming predictions into percentile ranks (`rankdata(p) / N`) preserves monotonic ordering while ensuring robust, threshold-invariant ensembling.

---

## Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Data Processing:** NumPy, Pandas, SciPy, PyArrow
* **Machine Learning:** Scikit-learn, LightGBM
* **APIs & Data Access:** NumerAPI, Kaggle API

```bash
# Clone the repository
git clone https://github.com/sachin23-an/quant-kaggle-research.git
cd quant-kaggle-research

# Install dependencies
pip install -r requirements.txt
```

---

## Author

**Sachin Kumar**  
* Indian Institute of Technology Madras (BS in Data Science)
* GitHub: [@sachin23-an](https://github.com/sachin23-an)
* Numerai Model: [`sachin23_an`](https://numer.ai/sachin23_an)
