"""
Step 3: Explainability layer using SHAP.
Global feature importance + individual (local) explanation.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load model + test data
# ---------------------------------------------------------
model = joblib.load("model_xgb.pkl")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# ---------------------------------------------------------
# 2. Build SHAP explainer (TreeExplainer is fast & exact for XGBoost)
# ---------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ---------------------------------------------------------
# 3. Global feature importance (summary plot)
# ---------------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved -> shap_summary.png (global feature importance)")

# Bar version (mean |SHAP value|) - easier to read for a report
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved -> shap_bar.png (ranked feature importance)")

# ---------------------------------------------------------
# 4. Local explanation for a few individual customers
#    - one predicted high risk, one predicted low risk
# ---------------------------------------------------------
probs = model.predict_proba(X_test)[:, 1]
high_risk_idx = int(np.argmax(probs))
low_risk_idx = int(np.argmin(probs))

for label, idx in [("HIGH_RISK", high_risk_idx), ("LOW_RISK", low_risk_idx)]:
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[idx], X_test.iloc[idx], show=False
    )
    plt.tight_layout()
    fname = f"shap_waterfall_{label}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {fname} (customer risk score: {probs[idx]:.2%})")

# ---------------------------------------------------------
# 5. Plain-English explanation generator (rule-based on top SHAP features)
# ---------------------------------------------------------
def explain_customer(idx, top_n=3):
    row_shap = shap_values[idx]
    feature_names = X_test.columns
    contrib = pd.Series(row_shap, index=feature_names).sort_values(key=abs, ascending=False)
    top_features = contrib.head(top_n)

    prob = probs[idx]
    verdict = "HIGH RISK of default" if prob >= 0.5 else "LOW RISK of default"

    lines = [f"Customer #{idx}: {verdict} (probability: {prob:.1%})", "Top contributing factors:"]
    for feat, val in top_features.items():
        direction = "increased" if val > 0 else "decreased"
        lines.append(f"  - {feat} = {X_test.iloc[idx][feat]:.0f}  ({direction} risk by {abs(val):.3f})")
    return "\n".join(lines)

print("\n" + "=" * 60)
print(explain_customer(high_risk_idx))
print("=" * 60)
print(explain_customer(low_risk_idx))
print("=" * 60)

# Save SHAP values for use in the app
np.save("shap_values.npy", shap_values)
joblib.dump(explainer.expected_value, "shap_expected_value.pkl")
print("\nSaved shap_values.npy, shap_expected_value.pkl")
