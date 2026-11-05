"""
Employee Attrition Prediction (Module 34) -- wraps the model trained in
attrition_model.ipynb. Output is a LOW/MEDIUM/HIGH risk indicator for HR review,
never grounds for automatic action (per the project report).
"""
import os
import joblib
import pandas as pd

_BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "models", "attrition_bundle.pkl")
_bundle = joblib.load(_BUNDLE_PATH)

model = _bundle["model"]
feature_columns = _bundle["feature_columns"]
categorical_cols = _bundle["categorical_cols"]
risk_thresholds = _bundle["risk_thresholds"]


def risk_bucket(probability: float) -> str:
    if probability < risk_thresholds["low_max"]:
        return "LOW"
    elif probability < risk_thresholds["medium_max"]:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_attrition_risk(employee: dict) -> dict:
    """employee: raw employee record (same shape as a row of the original CSV,
    minus Attrition/EmployeeCount/Over18/StandardHours/EmployeeNumber)."""
    row = pd.DataFrame([employee])
    row_encoded = pd.get_dummies(row, columns=[c for c in categorical_cols if c in row.columns])

    # Align columns exactly with training feature set (missing dummy cols = 0)
    for col in feature_columns:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[feature_columns].astype(float)

    probability = float(model.predict_proba(row_encoded)[0][1])
    return {
        "attrition_probability_pct": round(probability * 100, 2),
        "risk_level": risk_bucket(probability),
    }
