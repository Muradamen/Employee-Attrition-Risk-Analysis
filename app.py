import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Employee Attrition Predictor", page_icon="🏢", layout="wide"
)


# -------------------------------
# LOAD MODEL + DATA
# -------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("models/attrition_model.pkl")
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    reference = df.drop("Attrition", axis=1)
    return model, reference


model, reference_df = load_assets()

# -------------------------------
# HEADER
# -------------------------------
st.title("🏢 AI-Powered Employee Attrition Predictor")
st.markdown("### Predict Risk • Understand Causes • Take Action")

# -------------------------------
# SIDEBAR INPUTS (GROUPED)
# -------------------------------
st.sidebar.header("👤 Employee Profile")

with st.sidebar.expander("📊 Personal Info", expanded=True):
    age = st.slider("Age", 18, 60, 30)
    distance = st.slider("Distance From Home", 1, 30, 10)
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with st.sidebar.expander("💼 Job Info", expanded=True):
    job_role = st.selectbox("Job Role", reference_df["JobRole"].unique())
    overtime = st.selectbox("OverTime", ["No", "Yes"])
    work_life = st.slider("Work-Life Balance", 1, 4, 3)

with st.sidebar.expander("💰 Compensation", expanded=True):
    income = st.slider("Monthly Income", 1000, 20000, 5000)
    stock = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    hike = st.slider("Percent Salary Hike", 10, 25, 15)

with st.sidebar.expander("📈 Satisfaction", expanded=True):
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)
    env_sat = st.slider("Environment Satisfaction", 1, 4, 3)

# -------------------------------
# BUILD INPUT (SAFE METHOD)
# -------------------------------
input_df = reference_df.iloc[[0]].copy()

input_df["Age"] = age
input_df["DistanceFromHome"] = distance
input_df["MonthlyIncome"] = income
input_df["OverTime"] = overtime
input_df["StockOptionLevel"] = stock
input_df["PercentSalaryHike"] = hike
input_df["JobSatisfaction"] = job_sat
input_df["EnvironmentSatisfaction"] = env_sat
input_df["WorkLifeBalance"] = work_life
input_df["JobRole"] = job_role
input_df["MaritalStatus"] = marital

# -------------------------------
# MAIN LAYOUT
# -------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Employee Snapshot")
    st.dataframe(input_df)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Generate Analysis"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    with col2:
        st.subheader("📊 Prediction Result")

        if prob > 0.6:
            st.error(f"🔴 High Attrition Risk ({prob:.2%})")
        elif prob > 0.3:
            st.warning(f"🟡 Medium Risk ({prob:.2%})")
        else:
            st.success(f"🟢 Low Risk ({prob:.2%})")

    # -------------------------------
    # SHAP EXPLANATION
    # -------------------------------
    st.divider()
    st.subheader("🧠 Why is this happening? (Explainability)")

    try:
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
        st.warning("SHAP explanation not available.")
        st.text(str(e))

    # -------------------------------
    # BUSINESS INSIGHTS
    # -------------------------------
    st.divider()
    st.subheader("📋 HR Recommendations")

    if overtime == "Yes" and prob > 0.4:
        st.warning("⚠️ Employee is working overtime → Risk of burnout")

    if income < 4000:
        st.info("💡 Salary below competitive level → Consider adjustment")

    if work_life <= 2:
        st.warning("⚠️ Poor work-life balance → Improve flexibility")

    if job_sat <= 2:
        st.warning("⚠️ Low job satisfaction → Engagement intervention needed")
