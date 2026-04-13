import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Essential imports for unpickling the specific Pipeline used in your notebook
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# -------------------------------
# 1. CONFIG & ASSET LOADING
# -------------------------------
st.set_page_config(
    page_title="AI Employee Attrition Predictor", page_icon="🏢", layout="wide"
)


@st.cache_resource
def load_assets():
    try:
        # Using relative paths for Streamlit Cloud compatibility
        model = joblib.load("models/attrition_model.pkl")
        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        reference = df.drop("Attrition", axis=1)
        return model, reference, None
    except Exception as e:
        return None, None, str(e)


model, reference_df, error_msg = load_assets()

# -------------------------------
# 2. HEADER
# -------------------------------
st.title("🏢 AI-Powered Employee Attrition Predictor")
st.markdown("### Predict Risk • Understand Causes • Take Action")

if error_msg:
    st.error(f"🛑 Error loading assets: {error_msg}")
    st.info(
        "Ensure 'models/attrition_model.pkl' and the CSV file are in your GitHub repo."
    )
    st.stop()

# -------------------------------
# 3. SIDEBAR INPUTS (Defined FIRST)
# -------------------------------
st.sidebar.header("👤 Employee Profile")

with st.sidebar.expander("📊 Personal Info", expanded=True):
    # Variables are created here
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home", 1, 30, 10)
    income = st.number_input("Monthly Income ($)", value=5000)
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    stock_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)

# -------------------------------
# 4. DATA PREPARATION (Uses variables from Step 3)
# -------------------------------
# We start with a row of real data to ensure all 30+ columns exist
input_df = reference_df.iloc[0:1].copy()

# Update the columns with user input from the sidebar
input_df.at[0, "Age"] = age
input_df.at[0, "DistanceFromHome"] = distance
input_df.at[0, "MonthlyIncome"] = income
input_df.at[0, "OverTime"] = overtime
input_df.at[0, "JobSatisfaction"] = job_satisfaction
input_df.at[0, "StockOptionLevel"] = stock_level

# -------------------------------
# 5. PREDICTION & ANALYSIS
# -------------------------------
if st.button("🔍 Run Risk Analysis"):
    # Calculate Probability
    prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Probability", f"{prob:.1%}")

    with col2:
        if prob > 0.6:
            st.error("🔴 HIGH RISK: Intervention Recommended")
        elif prob > 0.3:
            st.warning("🟡 MEDIUM RISK: Monitor Engagement")
        else:
            st.success("🟢 LOW RISK: Stable")

    # -------------------------------
    # 6. SHAP EXPLANATION (Corrected Quotes)
    # -------------------------------
    st.divider()
    # Fixed SyntaxError: Using single quotes outside to allow double quotes inside
    st.subheader('🧠 Why is this happening? (AI "Why" Explanation)')

    try:
        # Accessing internal pipeline steps for SHAP
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
    except Exception as e:
        st.warning("SHAP visual is still generating or version mismatch detected.")
        st.caption(f"Technical detail: {e}")

    # -------------------------------
    # 7. BUSINESS INSIGHTS
    # -------------------------------
    st.divider()
    st.subheader("📋 Targeted HR Recommendations")

    recs = []
    if overtime == "Yes" and prob > 0.4:
        recs.append(
            "⚠️ **Burnout Risk:** High correlation between overtime and attrition for this profile."
        )
    if income < 4000:
        recs.append(
            "💰 **Compensation Review:** Monthly income is below the 40th percentile for similar roles."
        )
    if distance > 15:
        recs.append(
            "🚗 **Commute Stress:** Long distance detected; consider remote flexibility."
        )

    if recs:
        for r in recs:
            st.write(r)
    else:
        st.write("Current profile parameters show high organizational fit.")
