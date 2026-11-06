"""
routes/predict_routes.py
POST /api/predict — accept health parameters, return risk prediction + SHAP explanation.
"""

import json
import sys
import os

from flask import Blueprint, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.predict import predict, validate_input

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def make_prediction():
    """
    Endpoint: POST /api/predict

    Accepts JSON body with 8 health parameters.
    Returns diabetes risk prediction, probability, risk category, and SHAP values.

    Request Body:
        {
            "pregnancies": 2,
            "glucose": 148,
            "bloodPressure": 72,
            "skinThickness": 35,
            "insulin": 0,
            "bmi": 33.6,
            "diabetesPedigreeFunction": 0.627,
            "age": 50
        }

    Response (200):
        {
            "prediction": 1,
            "probability": 82.4,
            "risk_label": "High Risk",
            "risk_category": "high",
            "risk_color": "#dc2626",
            "shap_values": { "Glucose": 0.42, "BMI": 0.21, ... },
            "top_factors": ["Glucose", "BMI", "Age"],
            "recommendation": "...",
            "model_used": "XGBoost",
            "record_id": 7
        }
    """
    # Parse JSON body
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Server-side validation
    errors = validate_input(data)
    if errors:
        return jsonify({"error": "Validation failed.", "details": errors}), 422

    try:
        result = predict(data)

        # Persist prediction to database
        from app import db
        from models.db_models import PredictionHistory

        record = PredictionHistory(
            pregnancies        = float(data.get("pregnancies", 0)),
            glucose            = float(data["glucose"]),
            blood_pressure     = float(data["bloodPressure"]),
            skin_thickness     = float(data.get("skinThickness", 0)) if data.get("skinThickness") else None,
            insulin            = float(data.get("insulin", 0)) if data.get("insulin") else None,
            bmi                = float(data["bmi"]),
            diabetes_pedigree  = float(data["diabetesPedigreeFunction"]),
            age                = float(data["age"]),
            prediction         = result["prediction"],
            probability        = result["probability"],
            risk_label         = result["risk_label"],
            risk_category      = result["risk_category"],
            shap_json          = json.dumps(result["shap_values"]),
        )
        db.session.add(record)
        db.session.commit()

        result["record_id"] = record.id
        return jsonify(result), 200

    except RuntimeError as e:
        # Model files not found
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        from app import db
        db.session.rollback()
        return jsonify({"error": "Prediction failed.", "detail": str(e)}), 500

