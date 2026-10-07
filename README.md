# Classifier & Ensemble — Trees to Stacking on Telco Churn

Progression from a single decision tree through bagged trees (Random Forest), boosted trees (XGBoost), and a stacked ensemble — on the Telco Customer Churn dataset.

## Dataset
- `fetch_openml('telco-customer-churn', version=1)`
- 7,032 rows × 30 raw features (categorical + numeric mix; ID dropped)
- Target: binary Churn, imbalance ~27% positive
- Preprocessing: `TotalCharges` coerced to numeric, nulls dropped, categoricals one-hot encoded (drop_first=True)

## Approach
1. Stratified 80/20 split
2. Decision Tree baseline — unpruned (max overfit) and depth-limited (variance-controlled); `max_depth` swept 1–20
3. Random Forest — 300 trees, default feature subsampling
4. XGBoost — 500 rounds, lr=0.05, early stopping on validation log-loss
5. Stacking — RF + XGB + scaled Logistic Regression → Logistic Regression meta-learner, `cv=5` for out-of-fold meta-features
6. Confusion matrices + per-class classification reports across all models

## Results

### Test accuracy
| Model | Train | Test | Notes |
|---|---|---|---|
| Tree (unpruned, depth 24) | 0.9988 | 0.7264 | Pure memorization |
| Tree (depth 5) | 0.8046 | 0.7811 | Bias-variance balanced |
| Random Forest (300) | 0.9988 | 0.7868 | Variance cancels |
| XGBoost (early stop @ 66) | 0.8284 | 0.7953 | Best single model |
| **Stack (RF + XGB + LR)** | **0.8423** | **0.8010** | Best overall |

### Churn class performance (the class that matters)
| Model | Churn Precision | Churn Recall | Churn F1 |
|---|---|---|---|
| Tree (depth 5) | 0.605 | 0.508 | 0.552 |
| Random Forest | 0.625 | 0.495 | 0.552 |
| XGBoost | 0.643 | 0.516 | 0.573 |
| **Stack (RF + XGB + LR)** | **0.649** | **0.548** | **0.594** |

## Key Observations
- **Accuracy is a misleading metric here.** Dummy baseline (always predict "No Churn") scores ~0.73. Every model sits only 5–7 points above a classifier that does nothing. The metric that matters is **Churn recall** — every model misses ~45–50% of churners.
- **Unpruned tree demonstrates overfitting cleanly:** train 0.9988 / test 0.7264. Constraining depth to 5 recovers ~5 test points.
- **Bagging vs. boosting:** RF plateaus and never overfits (averaging cancels variance); XGBoost overfits without early stopping — best iteration at 66, well before the 500-round cap.
- **Stacking gains are marginal (+0.6% over XGBoost).** RF and XGB are correlated tree-based learners sharing the same features. The tiny gain comes from LR contributing a different error profile — not from the meta-learner being clever.
- **Scaling matters only for LR.** Trees split on rank order and ignore scale. The LR base learner and meta-learner were wrapped in `StandardScaler` pipelines to resolve convergence issues; this moved stack accuracy from 0.8003 → 0.8010.

## Stack
scikit-learn · xgboost · numpy · pandas · matplotlib

## Run
```bash
pip install scikit-learn xgboost numpy pandas matplotlib
python churn_trees.py