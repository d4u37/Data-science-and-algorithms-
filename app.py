"""
Streamlit app: Explainable Credit Card Default Risk Predictor
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Default Risk - Explainable AI", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model_xgb.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_data
def load_data():
    X_test = pd.read_csv("X_test.csv")
    return X_test

model, explainer = load_artifacts()
X_test = load_data()

st.title("💳 Explainable Credit Card Default Risk Predictor")
st.caption("Predicts default risk and explains *why*, using SHAP feature attributions.")

tab1, tab2 = st.tabs(["🔍 Pick an existing customer", "✏️ Enter a custom profile"])

FEATURES = list(X_test.columns)

def show_prediction(row_df):
    prob = model.predict_proba(row_df)[0, 1]
    verdict = "🔴 HIGH RISK of default" if prob >= 0.5 else "🟢 LOW RISK of default"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Default probability", f"{prob:.1%}")
        st.subheader(verdict)

    shap_vals = explainer.shap_values(row_df)
    contrib = pd.Series(shap_vals[0], index=FEATURES).sort_values(key=abs, ascending=False)
    top = contrib.head(5)

    with col2:
        st.subheader("Top factors driving this decision")
        for feat, val in top.items():
            direction = "⬆️ increases risk" if val > 0 else "⬇️ decreases risk"
            st.write(f"**{feat}** = {row_df.iloc[0][feat]:.0f} — {direction} (impact: {val:+.3f})")

    st.subheader("SHAP Waterfall Explanation")
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_vals[0], row_df.iloc[0], show=False
    )
    st.pyplot(fig)

with tab1:
    idx = st.selectbox("Select a customer index from the test set", X_test.index[:200])
    row_df = X_test.loc[[idx]]
    st.dataframe(row_df)
    if st.button("Explain this customer", key="existing"):
        show_prediction(row_df)

with tab2:
    st.write("Enter customer details:")
    c1, c2, c3 = st.columns(3)
    with c1:
        limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", 10000, 1000000, 200000, step=10000)
        sex = st.selectbox("Sex (1=Male, 2=Female)", [1, 2])
        education = st.selectbox("Education (1=Grad,2=Univ,3=HS,4=Other)", [1, 2, 3, 4])
        marriage = st.selectbox("Marriage (1=Married,2=Single,3=Other)", [1, 2, 3])
        age = st.number_input("Age", 18, 90, 35)
    with c2:
        pay_0 = st.slider("PAY_0 (most recent repayment status)", -2, 8, 0)
        pay_2 = st.slider("PAY_2", -2, 8, 0)
        pay_3 = st.slider("PAY_3", -2, 8, 0)
        pay_4 = st.slider("PAY_4", -2, 8, 0)
        pay_5 = st.slider("PAY_5", -2, 8, 0)
        pay_6 = st.slider("PAY_6", -2, 8, 0)
    with c3:
        bill_amt1 = st.number_input("Recent Bill Amount", 0, 1000000, 50000, step=1000)
        pay_amt1 = st.number_input("Recent Payment Amount", 0, 1000000, 20000, step=1000)

    if st.button("Predict & Explain", key="custom"):
        custom_row = {c: 0 for c in FEATURES}
        custom_row.update({
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
            "MARRIAGE": marriage, "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
            "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill_amt1, "PAY_AMT1": pay_amt1,
        })
        row_df = pd.DataFrame([custom_row])[FEATURES]
        show_prediction(row_df)

st.markdown("---")
st.caption("Model: XGBoost | Explainability: SHAP | Dataset: UCI Default of Credit Card Clients")
