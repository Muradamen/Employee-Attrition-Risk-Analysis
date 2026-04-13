import streamlit as st
import pandas as pd
import joblib
import os

# ✅ Required imports for loading model
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier


# -------------------------------
# LOAD MODEL + REFERENCE DATA
# -------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("models/attrition_model.pkl")

        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
        reference_df = df.drop("Attrition", axis=1)

        return model, reference_df

    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()


model_pipeline, reference_df = load_assets()


# -------------------------------
# UI
# -------------------------------
st.title("🏢 IBM HR Attrition Predictor")

st.sidebar.header("Employee Input")


# -------------------------------
# BUILD INPUTS FROM DATASET
# -------------------------------
input_data = {}

for col in reference_df.columns:

    if reference_df[col].dtype == "object":
        input_data[col] = st.sidebar.selectbox(
            col, options=reference_df[col].unique(), index=0
        )
    else:
        input_data[col] = st.sidebar.slider(
            col,
            int(reference_df[col].min()),
            int(reference_df[col].max()),
            int(reference_df[col].median()),
        )

input_df = pd.DataFrame([input_data])


# -------------------------------
# PREDICTION
# -------------------------------
st.subheader("Input Data")
st.write(input_df)

if st.button("Predict Attrition"):
    try:
        prediction = model_pipeline.predict(input_df)[0]
        prob = model_pipeline.predict_proba(input_df)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.error(f"⚠️ High Attrition Risk (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Attrition Risk (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"Prediction error: {e}")
