# Retrain the model with current sklearn version

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert target
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Drop useless columns
df = df.drop(["EmployeeNumber", "Over18", "StandardHours"], axis=1, errors="ignore")

# Features / target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Column types
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Pipelines
numeric_pipeline = Pipeline([("scaler", StandardScaler())])

categorical_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    [("num", numeric_pipeline, num_cols), ("cat", categorical_pipeline, cat_cols)]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model pipeline with SMOTE
xgb_smote_pipeline = ImbPipeline(
    [
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(eval_metric="logloss", random_state=42)),
    ]
)

# Hyperparameter tuning
param_grid_smote_xgb = {"model__n_estimators": [100, 200], "model__max_depth": [3, 6]}

grid_smote_xgb = GridSearchCV(
    xgb_smote_pipeline, param_grid_smote_xgb, cv=3, scoring="roc_auc", n_jobs=-1
)

print("Training model...")
grid_smote_xgb.fit(X_train, y_train)

best_model = grid_smote_xgb.best_estimator_
print("Best Params:", grid_smote_xgb.best_params_)

# Save model
joblib.dump(best_model, "models/attrition_model.pkl")
print("Model retrained and saved!")
