"""
ml/predict.py
Inference engine — loads trained models and runs predictions with SHAP explanations.

This module is the single source of truth for all ML inference.
Flask routes call predict() and get back a fully structured response.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocess import FEATURES

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR      = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "saved_models"))
PRIMARY_MODEL  = "XGBoost"   # Best ROC-AUC — used for the main prediction card

# Input validation ranges (clinically reasonable bounds)
VALIDATION_RULES = {
    "pregnancies":              {"min": 0,   "max": 20,  "required": True},
    "glucose":                  {"min": 50,  "max": 300, "required": True},
    "bloodPressure":            {"min": 30,  "max": 180, "required": True},
    "skinThickness":            {"min": 0,   "max": 100, "required": False},
    "insulin":                  {"min": 0,   "max": 900, "required": False},
    "bmi":                      {"min": 10,  "max": 80,  "required": True},
    "diabetesPedigreeFunction": {"min": 0.0, "max": 3.0, "required": True},
    "age":                      {"min": 1,   "max": 120, "required": True},
}

# Cached artifacts (loaded once at first inference, reused after)
_cache: dict = {}


def _load_artifacts() -> tuple:
    """
    Load and cache model, explainer, and preprocessor from disk.
    Uses a module-level cache so the files are only read once per server start.
    """
    global _cache
    if not _cache:
        try:
            _cache["model"]        = joblib.load(os.path.join(MODEL_DIR, f"{PRIMARY_MODEL}_model.pkl"))
            _cache["explainer"]    = joblib.load(os.path.join(MODEL_DIR, f"{PRIMARY_MODEL}_explainer.pkl"))
            _cache["preprocessor"] = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model files not found. Please run: python ml/train.py\nDetails: {e}"
            )
    return _cache["model"], _cache["explainer"], _cache["preprocessor"]


def validate_input(data: dict) -> list[str]:
    """
    Validate that input fields are present, numeric, and within clinical bounds.

    Args:
        data: Raw JSON dict from the API request.

    Returns:
        List of error strings. Empty list = valid input.
    """
    errors = []
    for field, rules in VALIDATION_RULES.items():
        value = data.get(field)

        if rules["required"] and value is None:
            errors.append(f"'{field}' is required.")
            continue

        if value is not None:
            try:
                val = float(value)
                if val < rules["min"] or val > rules["max"]:
                    errors.append(
                        f"'{field}' must be between {rules['min']} and {rules['max']}."
                    )
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be a numeric value.")

    return errors


def _get_risk_info(probability: float) -> dict:
    """
    Map a raw probability (0.0–1.0) to a human-readable risk category.

    Thresholds:
        < 35%  → Low Risk
        35–65% → Moderate Risk
        > 65%  → High Risk
    """
    if probability < 0.35:
        return {"label": "Low Risk",      "category": "low",      "color": "#16a34a"}
    elif probability < 0.65:
        return {"label": "Moderate Risk", "category": "moderate", "color": "#d97706"}
    else:
        return {"label": "High Risk",     "category": "high",     "color": "#dc2626"}


def _generate_recommendation(shap_dict: dict, category: str) -> str:
    """
    Generate a plain-language recommendation based on the top SHAP drivers.

    Args:
        shap_dict: Feature → SHAP value mapping.
        category:  'low', 'moderate', or 'high'.

    Returns:
        Recommendation string for display in the UI.
    """
    top_risk = [k for k, v in sorted(shap_dict.items(), key=lambda x: x[1], reverse=True) if v > 0][:3]
    factors  = ", ".join(top_risk) if top_risk else "your overall health profile"

    messages = {
        "high": (
            f"⚠️ High diabetes risk detected. Key contributing factors: {factors}. "
            "Please consult a healthcare professional as soon as possible for a formal evaluation and HbA1c test."
        ),
        "moderate": (
            f"⚠️ Moderate diabetes risk detected. Main factors: {factors}. "
            "Consider scheduling a check-up with your doctor, reviewing your diet and physical activity levels."
        ),
        "low": (
            "✅ Low diabetes risk. Keep maintaining a healthy lifestyle — balanced diet, "
            "regular exercise, and routine health check-ups are your best prevention strategy."
        ),
    }
    return messages.get(category, "Please consult a healthcare professional.")


def _extract_shap_values(explainer, input_scaled: np.ndarray) -> np.ndarray:
    """
    Extract SHAP values for the positive class (diabetes = 1).
    Handles both list output (binary classifiers) and array output.
    """
    shap_vals = explainer.shap_values(input_scaled)

    if isinstance(shap_vals, list):
        # Binary classification returns [class_0_vals, class_1_vals]
        return shap_vals[1][0]
    else:
        return shap_vals[0]


def predict(raw_input: dict) -> dict:
    """
    Full inference pipeline for a single patient.

    Steps:
        1. Map raw API keys to DataFrame column names
        2. Preprocess using the saved pipeline (impute + scale)
        3. Predict probability with the primary model (XGBoost)
        4. Compute SHAP values for explainability
        5. Return a fully structured response dict

    Args:
        raw_input: Validated dict from the API (camelCase keys from frontend).

    Returns:
        Structured prediction dict ready to be JSON-serialized.
    """
    model, explainer, preprocessor = _load_artifacts()

    # Map camelCase API keys → DataFrame column names (must match FEATURES order)
    input_df = pd.DataFrame([{
        "Pregnancies":              float(raw_input.get("pregnancies", 0)),
        "Glucose":                  float(raw_input["glucose"]),
        "BloodPressure":            float(raw_input["bloodPressure"]),
        "SkinThickness":            float(raw_input.get("skinThickness", 0)),
        "Insulin":                  float(raw_input.get("insulin", 0)),
        "BMI":                      float(raw_input["bmi"]),
        "DiabetesPedigreeFunction": float(raw_input["diabetesPedigreeFunction"]),
        "Age":                      float(raw_input["age"]),
    }], columns=FEATURES)

    # Preprocess — same pipeline fitted on training data
    input_scaled = preprocessor.transform(input_df)

    # Predict
    probability_raw  = float(model.predict_proba(input_scaled)[0][1])
    prediction       = int(probability_raw >= 0.5)
    risk_info        = _get_risk_info(probability_raw)
    probability_pct  = round(probability_raw * 100, 2)

    # SHAP explanation
    shap_array = _extract_shap_values(explainer, input_scaled)
    shap_dict  = {
        feat: round(float(val), 4)
        for feat, val in zip(FEATURES, shap_array)
    }

    # Sort features by absolute SHAP value (most influential first)
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    top_factors = [k for k, v in shap_dict.items() if v > 0][:3]

    recommendation = _generate_recommendation(shap_dict, risk_info["category"])

    return {
        "prediction":    prediction,
        "probability":   probability_pct,
        "risk_label":    risk_info["label"],
        "risk_category": risk_info["category"],
        "risk_color":    risk_info["color"],
        "shap_values":   shap_sorted,
        "top_factors":   top_factors,
        "recommendation": recommendation,
        "model_used":    PRIMARY_MODEL,
    }

