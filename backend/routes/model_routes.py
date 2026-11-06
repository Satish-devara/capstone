"""
routes/model_routes.py
GET /api/models — return evaluation metrics for all trained models.
"""

import json
import os
import sys

from flask import Blueprint, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_bp = Blueprint("models", __name__)

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "saved_models"))


@model_bp.route("/models", methods=["GET"])
def get_model_metrics():
    """
    Endpoint: GET /api/models

    Returns the evaluation metrics for all 5 trained ML models.
    Data comes from metrics_report.json generated during training.

    Response (200):
        {
            "LogisticRegression": { "accuracy": 0.76, "roc_auc": 0.83, ... },
            "RandomForest":       { "accuracy": 0.80, "roc_auc": 0.88, ... },
            "XGBoost":            { "accuracy": 0.82, "roc_auc": 0.90, ... },
            "SVM":                { "accuracy": 0.77, "roc_auc": 0.84, ... },
            "KNN":                { "accuracy": 0.74, "roc_auc": 0.80, ... }
        }
    """
    metrics_path = os.path.join(MODEL_DIR, "metrics_report.json")

    if not os.path.exists(metrics_path):
        return jsonify({
            "error": "No model metrics found. Please run: python ml/train.py"
        }), 404

    with open(metrics_path) as f:
        metrics = json.load(f)

    return jsonify(metrics), 200

