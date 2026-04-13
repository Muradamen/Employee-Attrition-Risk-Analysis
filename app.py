%%writefile app.py

import streamlit as st
import pandas as pd
import joblib
import os

# 🔥 ADD THESE (CRITICAL FIX for imblearn components)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 🔥 ADD THESE (CRITICAL FIX for sklearn and xgboost components)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

# @st.cache_resource is used to cache the model loading process
@st.cache_resource
def load_assets():
    # Load the trained model pipeline
    try:
        model_pipeline = joblib.load('models/attrition_model.pkl')
        return model_pipeline, None # Return model_pipeline and a placeholder for reference_df
    except FileNotFoundError:
        st.error("Model file 'models/attrition_model.pkl' not found. Please ensure it's in the correct path.")
        st.stop()

# Load assets (model) using the cached function
model_pipeline, reference_df = load_assets()
preprocessor = model_pipeline.named_steps['preprocessor']
model = model_pipeline.named_steps['model']

st.title("IBM HR Attrition Prediction Dashboard")
st.write("Predict employee attrition based on various factors.")

# Define input features and their types/ranges dynamically based on X from the notebook's kernel state
# In a standalone app, these would be hardcoded or loaded from a config.

# Helper function to get descriptive stats for numerical and unique values for categorical
def get_feature_info(feature_name, df_source):
    if df_source[feature_name].dtype == 'object': # Categorical
        return {'type': 'category', 'options': list(df_source[feature_name].unique()), 'default': df_source[feature_name].mode()[0]}
    else: # Numerical
        min_val = int(df_source[feature_name].min())
        max_val = int(df_source[feature_name].max())
        default_val = int(df_source[feature_name].median()) # Use median for default to be robust to outliers
        return {'type': 'number', 'min': min_val, 'max': max_val, 'default': default_val}

# Collect all feature names from X (assuming X is the global DataFrame from the notebook)
# Exclude 'EmployeeCount' if it somehow persisted and is constant
all_features = [col for col in X.columns if col not in ['EmployeeCount']]

feature_details_dynamic = {}
for col in all_features:
    info = get_feature_info(col, X)
    if info:
        feature_details_dynamic[col] = info

input_data = {}
st.sidebar.header("Employee Characteristics")

# Sort features alphabetically for consistent display in the sidebar
sorted_features = sorted(feature_details_dynamic.keys())

for feature in sorted_features:
    details = feature_details_dynamic[feature]
    if details['type'] == 'number':
        input_data[feature] = st.sidebar.slider(
            f"{feature}",
            min_value=details['min'],
            max_value=details['max'],
            value=details['default']
        )
    elif details['type'] == 'category':
        input_data[feature] = st.sidebar.selectbox(
            f"{feature}",
            options=details['options'],
            index=details['options'].index(details['default']) if details['default'] in details['options'] else 0
        )

input_df = pd.DataFrame([input_data])

st.subheader("Input Features for Prediction:")
st.write(input_df)

if st.button("Predict Attrition"):    # Check if the model_pipeline is loaded before making predictions
    if model_pipeline is not None:
        try:
            prediction = model_pipeline.predict(input_df)
            prediction_proba = model_pipeline.predict_proba(input_df)[:, 1]

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.error(f"**This employee is likely to attrite.** (Probability: {prediction_proba[0]:.2f})")
            else:
                st.success(f"**This employee is not likely to attrite.** (Probability: {prediction_proba[0]:.2f})")

            st.markdown(f"**Confidence (Probability of Attrition):** `{prediction_proba[0]:.2f}`")

            st.markdown("---")
            st.markdown("### Key factors influencing attrition (from model training):")
            # Assuming feat_imp is a global variable from the notebook's kernel state
            if 'feat_imp' in globals():
                top_features_list = feat_imp['Feature'].head(5).tolist()
                for i, feat in enumerate(top_features_list):
                    st.markdown(f"- {feat}")
            else:
                st.markdown("- Overtime")
                st.markdown("- Monthly Income")
                st.markdown("- Distance From Home")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model is not loaded. Cannot make predictions.")