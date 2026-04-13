import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. PAGE CONFIG
st.set_page_config(page_title="HR Attrition Predictor", page_icon="🏢", layout="wide")


# 2. LOAD DATA & MODEL
# Assuming your notebook exported the pipeline as 'attrition_model.pkl'
@st.cache_resource
def load_assets():
    model_pipeline = joblib.load("models/attrition_model.pkl")
    # We need a sample row to get the correct column structure
    sample_data = (
        pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        .iloc[0:1]
        .drop("Attrition", axis=1)
    )
    return model_pipeline, sample_data


model, reference_df = load_assets()

# 3. UI LAYOUT
st.title("🏢 Strategic Employee Retention AI")
st.markdown("### Predicting Flight Risk with XGBoost & SHAP Explainability")

with st.sidebar:
    st.header("👤 Employee Profile")
    # Key drivers identified in your notebook
    overtime = st.selectbox("Overtime", ["Yes", "No"], index=1)
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
    distance = st.slider("Distance From Home (km)", 1, 30, 5)
    age = st.slider("Employee Age", 18, 65, 30)
    stock_level = st.select_slider("Stock Option Level", options=[0, 1, 2, 3])
    satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

# 4. DATA PREPARATION
# We take the reference row and update it with user inputs
input_df = reference_df.copy()
input_df.at[0, "OverTime"] = overtime
input_df.at[0, "MonthlyIncome"] = monthly_income
input_df.at[0, "DistanceFromHome"] = distance
input_df.at[0, "Age"] = age
input_df.at[0, "StockOptionLevel"] = stock_level
input_df.at[0, "JobSatisfaction"] = satisfaction

# 5. PREDICTION
if st.button("Generate Risk Analysis"):
    # Probability prediction
    risk_prob = model.predict_proba(input_df)[0][1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Attrition Probability", f"{risk_prob:.1%}")

    status = (
        "🔴 HIGH RISK"
        if risk_prob > 0.6
        else "🟡 ATTENTION" if risk_prob > 0.3 else "🟢 STABLE"
    )
    col2.metric("Risk Status", status)

    # 6. SHAP EXPLANATION (The "Why")
    st.divider()
    st.subheader("🧠 Why is this employee at risk?")

    # Get the model and preprocessor from your pipeline
    model_obj = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    # Transform data for SHAP
    X_transformed = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model_obj)
    shap_values = explainer.shap_values(X_transformed)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.bar_plot(
        shap_values[0], feature_names=feature_names, max_display=10, show=False
    )
    st.pyplot(fig)

    # 7. BUSINESS RECOMMENDATIONS
    st.subheader("📋 HR Action Plan")
    if overtime == "Yes" and risk_prob > 0.4:
        st.warning(
            "⚠️ **Overtime Burnout:** This employee is working significant overtime. Recommend a workload review."
        )
    if monthly_income < 4000:
        st.info(
            "💡 **Compensation Gap:** Income is below industry average for this role. Consider a retention bonus."
        )
