"""
Step 2: Train baseline (Logistic Regression) and advanced (XGBoost) models,
handling class imbalance with SMOTE, and compare with proper metrics.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ---------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------
df = pd.read_csv("data_cleaned.csv")
X = df.drop(columns=["DEFAULT"])
y = df["DEFAULT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 2. Scale numeric features (needed for Logistic Regression)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. Handle class imbalance with SMOTE (on training data only)
# ---------------------------------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
X_train_scaled_res, y_train_scaled_res = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE :", y_train_res.value_counts().to_dict())

results = {}

# ---------------------------------------------------------
# 4. Baseline: Logistic Regression
# ---------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled_res, y_train_scaled_res)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = log_reg.predict(X_test_scaled)

results["Logistic Regression"] = {
    "model": log_reg,
    "y_prob": y_prob_lr,
    "y_pred": y_pred_lr,
    "auc": roc_auc_score(y_test, y_prob_lr),
}

# ---------------------------------------------------------
# 5. Random Forest
# ---------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)

results["Random Forest"] = {
    "model": rf,
    "y_prob": y_prob_rf,
    "y_pred": y_pred_rf,
    "auc": roc_auc_score(y_test, y_prob_rf),
}

# ---------------------------------------------------------
# 6. XGBoost (typically best performer on this dataset)
# ---------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=42, n_jobs=-1
)
xgb_model.fit(X_train_res, y_train_res)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = xgb_model.predict(X_test)

results["XGBoost"] = {
    "model": xgb_model,
    "y_prob": y_prob_xgb,
    "y_pred": y_pred_xgb,
    "auc": roc_auc_score(y_test, y_prob_xgb),
}

# ---------------------------------------------------------
# 7. Compare models
# ---------------------------------------------------------
print("\n===== Model Comparison (AUC-ROC) =====")
for name, res in results.items():
    print(f"{name:20s} AUC: {res['auc']:.4f}")
    print(classification_report(y_test, res["y_pred"], target_names=["No Default", "Default"]))

# ROC curve plot
plt.figure(figsize=(7, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=120)
print("\nSaved -> roc_comparison.png")

# Confusion matrix for best model (XGBoost)
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\nBest model: {best_name} (AUC={best['auc']:.4f})")

cm = confusion_matrix(y_test, best["y_pred"])
disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("Saved -> confusion_matrix.png")

# ---------------------------------------------------------
# 8. Save artifacts for the explainability step & app
# ---------------------------------------------------------
joblib.dump(xgb_model, "model_xgb.pkl")
joblib.dump(scaler, "scaler.pkl")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
X_train_res.to_csv("X_train_res.csv", index=False)

print("\nSaved model_xgb.pkl, scaler.pkl, X_test.csv, y_test.csv, X_train_res.csv")
