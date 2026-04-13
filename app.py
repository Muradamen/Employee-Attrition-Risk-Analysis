import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 🔥 IMPORTANT: If you used ANY custom transformer, import it here
# Example:
# from utils import CustomTransformer

# 1. PAGE CONFIG
st.set_page_config(page_title="HR Attrition Predictor", page_icon="🏢", layout="wide")


# 2. LOAD DATA & MODEL
@st.cache_resource
def load_assets():
    try:
        # ✅ Check if model file exists
        model_path = "models/attrition_model.pkl"
        data_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

        if not os.path.exists(model_path):
            st.error("❌ Model file not found. Check 'models/attrition_model.pkl'")
            st.stop()

        if not os.path.exists(data_path):
            st.error("❌ Dataset file not found.")
            st.stop()

        # ✅ Load model
        model_pipeline = joblib.load(model_path)

        # ✅ Load reference row
        sample_data = pd.read_csv(data_path).iloc[0:1].drop("Attrition", axis=1)

        return model_pipeline, sample_data

    except Exception as e:
        st.error("🚨 Error loading model:")
        st.code(str(e))
        st.stop()


model, reference_df = load_assets()


# 3. UI LAYOUT
st.title("🏢 Strategic Employee Retention AI")
st.markdown("### Predicting Flight Risk with XGBoost & SHAP Explainability")

with st.sidebar:
    st.header("👤 Employee Profile")

    overtime = st.selectbox("Overtime", ["Yes", "No"], index=1)
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
    distance = st.slider("Distance From Home (km)", 1, 30, 5)
    age = st.slider("Employee Age", 18, 65, 30)
    stock_level = st.select_slider("Stock Option Level", options=[0, 1, 2, 3])
    satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)


# 4. DATA PREPARATION
input_df = reference_df.copy()

input_df.at[0, "OverTime"] = overtime
input_df.at[0, "MonthlyIncome"] = monthly_income
input_df.at[0, "DistanceFromHome"] = distance
input_df.at[0, "Age"] = age
input_df.at[0, "StockOptionLevel"] = stock_level
input_df.at[0, "JobSatisfaction"] = satisfaction


# 5. PREDICTION
if st.button("Generate Risk Analysis"):

    try:
        risk_prob = model.predict_proba(input_df)[0][1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Attrition Probability", f"{risk_prob:.1%}")

        status = (
            "🔴 HIGH RISK"
            if risk_prob > 0.6
            else "🟡 ATTENTION" if risk_prob > 0.3 else "🟢 STABLE"
        )
        col2.metric("Risk Status", status)

        # 6. SHAP EXPLANATION
        st.divider()
        st.subheader("🧠 Why is this employee at risk?")

        model_obj = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]

        X_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 4))
        shap.bar_plot(
            shap_values[0], feature_names=feature_names, max_display=10, show=False
        )
        st.pyplot(fig)

        # 7. BUSINESS RECOMMENDATIONS
        st.subheader("📋 HR Action Plan")

        if overtime == "Yes" and risk_prob > 0.4:
            st.warning("⚠️ Overtime Burnout: Recommend workload review.")

        if monthly_income < 4000:
            st.info("💡 Compensation Gap: Consider salary adjustment.")

    except Exception as e:
        st.error("🚨 Prediction failed:")
        st.code(str(e))
