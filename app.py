import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Attrition AI Dashboard",
    page_icon="💼",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/attrition_model.pkl")

# Extract components
preprocessor = model.named_steps['preprocessor']
model_only = model.named_steps['model']

# -----------------------------
# HEADER
# -----------------------------
st.title("💼 Employee Attrition AI Dashboard")
st.markdown("Predict and explain employee attrition risk")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🧾 Employee Profile")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home", 1, 50, 10)
overtime = st.sidebar.selectbox("Overtime", ["Yes", "No"])
job_level = st.sidebar.slider("Job Level", 1, 5, 2)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)

# -----------------------------
# INPUT DATA
# -----------------------------
input_df = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "DistanceFromHome": [distance],
    "OverTime": [overtime],
    "JobLevel": [job_level],
    "YearsAtCompany": [years_at_company]
})

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Analyze Employee"):

    # Transform input
    input_transformed = preprocessor.transform(input_df)

    # Prediction
    prob = model_only.predict_proba(input_transformed)[0][1]

    # -----------------------------
    # DISPLAY METRICS
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.metric("Attrition Risk %", f"{prob*100:.2f}%")

    if prob > 0.7:
        risk = "🔴 High Risk"
        advice = "Immediate HR action required"
    elif prob > 0.4:
        risk = "🟡 Medium Risk"
        advice = "Monitor employee closely"
    else:
        risk = "🟢 Low Risk"
        advice = "Stable employee"

    col2.metric("Risk Level", risk)

    st.info(f"💡 Recommendation: {advice}")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.markdown("## 🧠 AI Explanation (Why this prediction?)")

    explainer = shap.Explainer(model_only)
    shap_values = explainer(input_transformed)

    # Waterfall plot (BEST for single prediction)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # -----------------------------
    # FEATURE IMPACT TABLE
    # -----------------------------
    st.markdown("## 📊 Feature Impact")

    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": shap_values.values[0]
    }).sort_values(by="Impact", key=abs, ascending=False)

    st.dataframe(shap_df.head(10))

    # -----------------------------
    # BUSINESS INTERPRETATION
    # -----------------------------
    st.markdown("## 📌 Key Risk Drivers")

    top_features = shap_df.head(3)["Feature"].values

    for f in top_features:
        st.warning(f"⚠️ {f} is strongly influencing attrition risk")