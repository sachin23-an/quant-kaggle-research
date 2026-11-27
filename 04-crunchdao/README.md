# DataCrunch 2: Equity Market Neutral Tournament Alpha Engine

Systematic quantitative forecasting engine for **DataCrunch #2** on **CrunchDAO**, predicting forward cross-sectional returns across the 3,000 most liquid US equities (Russell 3000 universe).

Developed by **Sachin Kumar** ([GitHub: @sachin23-an](https://github.com/sachin23-an))  
*BS in Data Science, Indian Institute of Technology Madras ('27)*  
*Model Name:* [`sachin23an`](https://hub.crunchdao.com/competitions/datacrunch-2/projects/15021/sachin23an)

---

## Tournament Overview

- **Platform:** [CrunchDAO](https://crunchdao.com/) / [DataCrunch #2](https://hub.crunchdao.com/competitions/datacrunch-2)
- **Prize Pool:** 1,000 $USDC / week
- **Universe:** 3,000 liquid US equities
- **Dataset Structure:**
  - `moon`: Weekly era identifier (772 historical training moons, 9 local evaluation test moons, live prediction rounds)
  - `id`: Asset ticker identifier (~2,100 assets per moon)
  - `Feature_1` .. `Feature_1150`: 1,150 continuous quantitative signals
  - `target`: Continuous forward relative equity return
- **Evaluation Metric:** Cross-sectional Pearson correlation per `moon` (Numerai era-correlation paradigm).

---

## Architecture & Implementation

1. **Regularized Linear / Ridge Factor Weighting:**
   - Fits $L_2$-regularized regression (`alpha=100.0`, zero-intercept) over expanding historical windows.
   - Learns linear weights across 1,150 features without collinearity explosion.
2. **Cross-Sectional Rank Transformation (Numerai-Style):**
   - Maps raw model predictions into uniform contemporaneous percentile ranks per `moon`:
     $$R_t(x_i) = \frac{\text{rank}(x_{i, t})}{N_t}$$
   - Guarantees strict scale invariance, robust quantile spreads, and immunity to cross-period volatility regime shifts.
3. **Execution Interface:**
   - `train(X_train, y_train, model_directory_path)`: Retrains model and dumps serialization artifacts to `resources/model.joblib`.
   - `infer(X_test, model_directory_path)`: Loads serialized model and generates rank-normalized cross-sectional predictions.

---

## Local Validation Results (`crunch test`)

Evaluated locally on held-out test moons ($773 \dots 781$):

| Metric | Baseline Score |
| :--- | :---: |
| **Mean Pearson Correlation** | **`+0.01927`** |
| **Correlation Standard Deviation** | `0.01923` |
| **Correlation Sharpe ($IR$)** | **`1.0019`** |
| **Positive Moon Ratio** | **`77.8%`** (7 / 9 positive moons) |
| **Max Single-Moon Correlation** | `+0.04532` |
| **Determinism Check** | **PASSED** |

---

## Operational Commands

```bash
# Setup environment
pip install crunch-cli --upgrade

# Test locally across evaluation moons
crunch test

# Submit code and model artifact
crunch push -m "Baseline regularized Ridge with cross-sectional percentile ranking"
```
