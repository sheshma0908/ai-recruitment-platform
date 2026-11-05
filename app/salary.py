"""
Salary Prediction (Module 35) -- wraps the quantile regression models trained
in salary_model.ipynb. Returns a RANGE, matching the report's example
(e.g. ₹7 LPA - ₹10 LPA), not a single point estimate.

Like the matching engine, the three XGBoost regressors are loaded from their
native .ubj format rather than pickle, to avoid cross-machine deserialization
issues with pickled XGBoost boosters.
"""
import os
import joblib
import pandas as pd
import xgboost as xgb

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_META_PATH = os.path.join(_MODELS_DIR, "salary_meta.pkl")

_meta = joblib.load(_META_PATH)
feature_columns = _meta["feature_columns"]
categorical_cols = _meta["categorical_cols"]
experience_map = _meta["experience_map"]

lower_model = xgb.XGBRegressor()
lower_model.load_model(os.path.join(_MODELS_DIR, "salary_lower_model.ubj"))

upper_model = xgb.XGBRegressor()
upper_model.load_model(os.path.join(_MODELS_DIR, "salary_upper_model.ubj"))


def predict_salary_range(job_posting: dict) -> dict:
    """job_posting: raw job posting fields (same shape as a row of the
    India job market CSV)."""
    row = pd.DataFrame([job_posting])
    row["experience_level_num"] = experience_map.get(job_posting.get("Experience_Level"), 2)
    row["skill_count"] = len(str(job_posting.get("Skills_Required", "")).split(","))

    row_encoded = pd.get_dummies(row, columns=[c for c in categorical_cols if c in row.columns])
    for col in feature_columns:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[feature_columns].astype(float)

    low = float(lower_model.predict(row_encoded)[0])
    high = float(upper_model.predict(row_encoded)[0])
    low, high = max(low, 0), max(high, low)

    return {
        "estimated_range_lpa": f"₹{low:.1f} LPA - ₹{high:.1f} LPA",
        "lower_bound_lpa": round(low, 1),
        "upper_bound_lpa": round(high, 1),
    }
