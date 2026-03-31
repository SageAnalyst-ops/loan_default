import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained artifacts
# -----------------------------
log_reg = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# Page Config and Title
# -----------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="centered")

# Title with logo
st.markdown(
    """
    <div style='display:flex; align-items:center'>
        <img src='https://img.icons8.com/color/48/000000/money-bag.png' width='50' style='margin-right:10px'/>
        <h1 style='color:#4B0082'>Loan Default Prediction System</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("Enter applicant details to predict loan default risk.")

# -----------------------------
# Model selection and threshold
# -----------------------------
model_choice = st.selectbox(
    "Select Prediction Model", 
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

threshold = st.slider(
    "Decision Threshold (Default Cut-off)",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)
st.caption("Applicants with probability above the threshold are classified as defaulters.")

# -----------------------------
# User Inputs with Illustrations
# -----------------------------
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧑 Age", 21, 70, 35)
    annual_income = st.number_input("💰 Annual Income", 5000, 200000, 45000)
    credit_score = st.number_input("📊 Credit Score", 300, 850, 650)
    loan_amount = st.number_input("🏦 Loan Amount", 500, 100000, 15000)
    loan_term_months = st.selectbox("📅 Loan Term (Months)", [12, 24, 36, 48, 60])

with col2:
    interest_rate = st.number_input("📈 Interest Rate (%)", 1.0, 40.0, 12.0)
    existing_loans_count = st.number_input("📋 Existing Loans Count", 0, 10, 1)
    debt_to_income_ratio = st.slider("📉 Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    missed_payments_last_12m = st.number_input("⚠️ Missed Payments (Last 12 Months)", 0, 10, 0)
    collateral_value = st.number_input("🛡 Collateral Value", 0, 200000, 20000)

# Categorical inputs
gender = st.selectbox("👤 Gender", ["Male", "Female"])
employment_status = st.selectbox(
    "💼 Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)
loan_purpose = st.selectbox(
    "🎯 Loan Purpose",
    ["Personal", "Business", "Education"]
)

# -----------------------------
# Build input feature template
# -----------------------------
input_data = pd.DataFrame(0, index=[0], columns=feature_names)

# Fill numerical features
input_data["age"] = age
input_data["annual_income"] = annual_income
input_data["credit_score"] = credit_score
input_data["loan_amount"] = loan_amount
input_data["loan_term_months"] = loan_term_months
input_data["interest_rate"] = interest_rate
input_data["existing_loans_count"] = existing_loans_count
input_data["debt_to_income_ratio"] = debt_to_income_ratio
input_data["missed_payments_last_12m"] = missed_payments_last_12m
input_data["collateral_value"] = collateral_value

# Fill categorical (one-hot encoded) features
input_data[f"gender_{gender}"] = 1
input_data[f"employment_status_{employment_status}"] = 1
input_data[f"loan_purpose_{loan_purpose}"] = 1

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Default"):

    # Scaling only for Logistic Regression
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_data)
        probability = log_reg.predict_proba(input_scaled)[0][1]

    elif model_choice == "Random Forest":
        probability = rf_model.predict_proba(input_data)[0][1]

    else:  # XGBoost
        probability = xgb_model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    # Risk Category
    if probability < 0.30:
        risk = "Low Risk"
        st.success("✅ Low Risk: Likely to Repay")

    elif probability < 0.60:
        risk = "Medium Risk"
        st.warning("⚠️ Medium Risk: Requires Further Review")

    else:
        risk = "High Risk"
        st.error("❌ High Risk: Likely to Default")

    # Display Results
    st.subheader("Prediction Results")
    st.markdown(f"**Selected Model:** {model_choice}")
    st.markdown(f"**Probability of Default:** {probability:.2f}")
    st.markdown(f"**Decision Threshold:** {threshold:.2f}")
    st.markdown(f"**Predicted Class:** {'Default' if prediction == 1 else 'Non-Default'}")
    st.markdown(f"**Risk Category:** {risk}")