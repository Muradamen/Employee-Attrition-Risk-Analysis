import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Explicit imports to help joblib unpickle the pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Employee Attrition Predictor", page_icon="🏢", layout="wide"
)


# -------------------------------
# LOAD ASSETS (FIXED)
# -------------------------------
@st.cache_resource
def load_assets():
    # Use relative paths for Streamlit Cloud compatibility
    model_path = os.path.join("models", "attrition_model.pkl")
    data_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        reference = df.drop("Attrition", axis=1)
        return model, reference
    except Exception as e:
        # We return None so the app can display a friendly error instead of a crash
        return None, str(e)


# Initialize
model, data_status = load_assets()

if model is None:
    st.error(f"🛑 Failed to load model. Error: {data_status}")
    st.info(
        "Check if 'models/attrition_model.pkl' exists and matches your scikit-learn version (1.3.2)."
    )
    st.stop()

reference_df = data_status  # If model loaded, data_status is our reference_df

# -------------------------------
# UI & INPUTS (KEEP YOUR SIDEBAR AS IS)
# -------------------------------
st.title("🏢 AI-Powered Employee Attrition Predictor")

# ... [Your Sidebar Code for Sliders goes here] ...

# -------------------------------
# PREDICTION LOGIC (REFINED)
# -------------------------------
# Ensure you are creating input_df by updating a copy of reference_df
# This ensures all 30+ columns exist even if you only have 6 sliders
input_df = reference_df.iloc[0:1].copy()

# Update only the features you have sliders for:
input_df.at[0, "Age"] = age
input_df.at[0, "DistanceFromHome"] = distance
input_df.at[0, "MonthlyIncome"] = income
input_df.at[0, "OverTime"] = overtime
input_df.at[0, "JobSatisfaction"] = job_satisfaction
input_df.at[0, "StockOptionLevel"] = stock_level

if st.button("🔍 Run Risk Analysis"):
    # Probability prediction
    prob = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Probability", f"{prob:.1%}")
    with col2:
        if prob > 0.6:
            st.error("🔴 HIGH RISK: Action Required")
        elif prob > 0.3:
            st.warning("🟡 MEDIUM RISK: Monitor Closely")
        else:
            st.success("🟢 LOW RISK: Stable")

    # -------------------------------
    # SHAP EXPLANATION (MATCHING PRODUCTION PIPELINE)
    # -------------------------------
    st.divider()
    st.subheader('🧠 AI "Why": What is driving this risk?')
    try:
        # Access steps from your specific Production Pipeline
        model_obj = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]

        # Transform input for the model
        X_transformed = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 4))
        # Use waterfall or bar plot
        shap.plots.bar(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_transformed[0],
                feature_names=feature_names,
            ),
            max_display=10,
            show=False,
        )
        st.pyplot(fig)
    except Exception as e:
        st.info("Visual explanation is generating...")
        # Fallback to simple bar if the complex one fails
        st.write(f"Detailed diagnostics: {e}")
