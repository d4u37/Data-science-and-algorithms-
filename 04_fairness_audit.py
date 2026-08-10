"""
Step 5: Fairness / bias audit.
Checks whether the model's error rates and predictions differ across
SEX and EDUCATION groups - a key concern in real-world credit risk models.
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 1. Load model + test data
# ---------------------------------------------------------
model = joblib.load("model_xgb.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Maps for readability
SEX_MAP = {1: "Male", 2: "Female"}
EDU_MAP = {1: "Grad School", 2: "University", 3: "High School", 4: "Other"}

results = pd.DataFrame({
    "SEX": X_test["SEX"].map(SEX_MAP),
    "EDUCATION": X_test["EDUCATION"].map(EDU_MAP),
    "y_true": y_test.values,
    "y_pred": y_pred,
    "y_prob": y_prob,
})

# ---------------------------------------------------------
# 2. Metric helper - per group
# ---------------------------------------------------------
def group_metrics(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        cm = confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Selection rate = % predicted as default (positive)
        selection_rate = (sub["y_pred"] == 1).mean()
        # False Positive Rate = predicted default, but actually did NOT default
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        # False Negative Rate = predicted no-default, but actually DID default
        fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
        # Recall / TPR = of actual defaulters, how many we caught
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        accuracy = (sub["y_true"] == sub["y_pred"]).mean()
        try:
            auc = roc_auc_score(sub["y_true"], sub["y_prob"])
        except ValueError:
            auc = np.nan

        rows.append({
            "group": g, "n": len(sub),
            "actual_default_rate": sub["y_true"].mean(),
            "selection_rate (predicted default %)": selection_rate,
            "accuracy": accuracy, "AUC": auc,
            "FPR (false alarm on good customers)": fpr,
            "FNR (missed actual defaulters)": fnr,
            "TPR (recall on defaulters)": tpr,
        })
    return pd.DataFrame(rows).set_index("group")

sex_metrics = group_metrics(results, "SEX")
edu_metrics = group_metrics(results, "EDUCATION")

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print("=" * 70)
print("FAIRNESS AUDIT: SEX")
print("=" * 70)
print(sex_metrics)

print("\n" + "=" * 70)
print("FAIRNESS AUDIT: EDUCATION")
print("=" * 70)
print(edu_metrics)

# ---------------------------------------------------------
# 3. Disparity summary (max gap between groups)
# ---------------------------------------------------------
def disparity_summary(metrics_df, name):
    print(f"\n--- Disparity in {name} ---")
    for col in ["selection_rate (predicted default %)", "FPR (false alarm on good customers)",
                "FNR (missed actual defaulters)", "AUC"]:
        gap = metrics_df[col].max() - metrics_df[col].min()
        print(f"  {col:45s} gap = {gap:.3f}")

disparity_summary(sex_metrics, "SEX")
disparity_summary(edu_metrics, "EDUCATION")

# ---------------------------------------------------------
# 4. Visualize
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sex_metrics["selection_rate (predicted default %)"].plot(
    kind="bar", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Predicted Default Rate by SEX")
axes[0, 0].set_ylabel("Selection Rate")
axes[0, 0].tick_params(axis='x', rotation=0)

sex_metrics[["FPR (false alarm on good customers)", "FNR (missed actual defaulters)"]].plot(
    kind="bar", ax=axes[0, 1])
axes[0, 1].set_title("Error Rates by SEX")
axes[0, 1].tick_params(axis='x', rotation=0)

edu_metrics["selection_rate (predicted default %)"].plot(
    kind="bar", ax=axes[1, 0], color="#DD8452")
axes[1, 0].set_title("Predicted Default Rate by EDUCATION")
axes[1, 0].set_ylabel("Selection Rate")
axes[1, 0].tick_params(axis='x', rotation=20)

edu_metrics[["FPR (false alarm on good customers)", "FNR (missed actual defaulters)"]].plot(
    kind="bar", ax=axes[1, 1])
axes[1, 1].set_title("Error Rates by EDUCATION")
axes[1, 1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig("fairness_audit.png", dpi=120)
print("\nSaved -> fairness_audit.png")

# ---------------------------------------------------------
# 5. Save tables
# ---------------------------------------------------------
sex_metrics.to_csv("fairness_sex_metrics.csv")
edu_metrics.to_csv("fairness_education_metrics.csv")
print("Saved -> fairness_sex_metrics.csv, fairness_education_metrics.csv")
