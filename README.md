# Explainable Credit Card Default Prediction

Predicts whether a credit card customer will default next month, and explains *why* using SHAP — mirrors the structure of an "Explainable Credit Risk" project (loan approval style) but applied to credit card default data.

## Dataset
UCI **"Default of Credit Card Clients"** dataset — 30,000 customers, Taiwan, 2005.
Source used here: `data.csv` (mirrored from a public GitHub repo of the UCI dataset).

Columns include: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`,
repayment status for last 6 months (`PAY_0..PAY_6`),
bill amounts (`BILL_AMT1..6`), payment amounts (`PAY_AMT1..6`),
and target `DEFAULT` (1 = defaulted next month).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_eda_preprocessing.py` | Loads data, cleans messy categories, runs EDA, saves `data_cleaned.csv` + `eda_overview.png` |
| 2 | `02_modeling.py` | Splits data, handles class imbalance with SMOTE, trains Logistic Regression / Random Forest / XGBoost, compares AUC-ROC, saves best model |
| 3 | `03_explainability.py` | SHAP global feature importance + per-customer waterfall explanations |
| 4 | `app.py` | Streamlit dashboard: pick a customer or enter a custom profile, get a risk score + plain-English explanation |
| 5 | `04_fairness_audit.py` | Bias/fairness audit — compares error rates across `SEX` and `EDUCATION` groups, saves `fairness_audit.png` + CSVs |

## Results (this run)
- Class balance: **22.1%** default rate (imbalanced — handled with SMOTE)
- Best model: **Random Forest**, AUC-ROC ≈ **0.757** (XGBoost close behind at ≈0.745)
- Top risk driver by far: **PAY_0** (most recent repayment status) — being 1+ months late dominates the prediction, consistent with published benchmarks on this dataset.

## Fairness audit findings

The model was checked for disparities in error rates across `SEX` and `EDUCATION` — this matters because a credit model that's more likely to wrongly deny (or wrongly approve) one group over another is a real regulatory and ethical risk, independent of overall accuracy.

**By SEX** (n=3598 Female, 2402 Male in test set):

| Metric | Female | Male | Gap |
|---|---|---|---|
| Actual default rate | 21.3% | 23.4% | 2.1 pts |
| Predicted default rate (selection rate) | 17.4% | 31.6% | **14.3 pts** |
| False Positive Rate (good customers wrongly flagged) | 10.1% | 23.1% | **13.1 pts** |
| False Negative Rate (defaulters missed) | 55.6% | 40.5% | 15.2 pts |
| AUC | 0.753 | 0.736 | 1.7 pts |

Even though actual default rates are almost the same between men and women (~21% vs ~23%), the model flags men as high-risk **nearly twice as often** as women, and men are more than **2x as likely to be a false alarm** (flagged as risky despite not defaulting). AUC is similar for both, so the model discriminates default risk about equally well within each group — but the *decision threshold* produces very different outcomes by sex. This is a classic case where overall accuracy hides a fairness problem.

**By EDUCATION**:

| Metric | Grad School | University | High School | Other |
|---|---|---|---|---|
| n | 2130 | 2774 | 1014 | 82 |
| Actual default rate | 19.0% | 23.6% | 25.7% | 6.1% |
| Predicted default rate | 22.7% | 25.9% | 18.0% | 1.2% |
| FPR | 15.7% | 17.1% | 10.4% | 1.3% |
| FNR | 47.4% | 45.7% | 59.8% | 100.0% |
| AUC | 0.756 | 0.754 | 0.713 | 0.592 |

The `Other` education group (n=82, a small slice of the data) is a red flag — the model catches **zero** actual defaulters in that group (FNR = 100%) and AUC drops to 0.59 (barely better than random). This is very likely a data-sparsity problem: too few examples for the model to learn that group's pattern reliably, not necessarily malicious bias. High School students also see a notably worse recall on actual defaulters (59.8% missed vs ~46% for Grad/University).

**Takeaway:** overall AUC (~0.75) looks solid and masks two separate issues — a threshold-driven disparity by sex, and a data-sparsity/reliability issue for underrepresented education groups. Both are worth flagging explicitly in a write-up; it's exactly the kind of finding that distinguishes a "built a model" project from a "understands responsible ML" project.

**Possible next steps if you want to take this further:**
- Try group-specific thresholds (equalized odds) or reweighing to close the sex gap
- Investigate whether `SEX` or a correlated feature is doing the work — try training a model without `SEX` and see if the gap persists (proxy discrimination)
- Collect/oversample more `Other`-education examples, or explicitly flag low-confidence predictions for that group instead of auto-deciding

## How to run

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn joblib streamlit

python 01_eda_preprocessing.py
python 02_modeling.py
python 03_explainability.py
python 04_fairness_audit.py
streamlit run app.py
```

## Files produced
- `eda_overview.png` — 4-panel EDA summary
- `roc_comparison.png`, `confusion_matrix.png` — model evaluation
- `shap_summary.png`, `shap_bar.png` — global explainability
- `shap_waterfall_HIGH_RISK.png`, `shap_waterfall_LOW_RISK.png` — example local explanations
- `model_xgb.pkl`, `scaler.pkl` — trained artifacts used by the app
- `fairness_audit.png`, `fairness_sex_metrics.csv`, `fairness_education_metrics.csv` — bias audit results

## Notes / next steps
- Threshold tuning (currently 0.5) could be adjusted based on business cost of false negatives vs false positives.
- See Fairness audit findings above for the sex/education disparities uncovered and possible mitigations.
