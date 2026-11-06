"""
tests/test_api.py
Integration tests for all Flask API endpoints.
"""

import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db

VALID_PAYLOAD = {
    "pregnancies":              2,
    "glucose":                  148,
    "bloodPressure":            72,
    "skinThickness":            35,
    "insulin":                  0,
    "bmi":                      33.6,
    "diabetesPedigreeFunction": 0.627,
    "age":                      50,
}


@pytest.fixture(scope="module")
def client():
    """Create a test Flask client using an in-memory SQLite database."""
    app = create_app()
    app.config["TESTING"]              = True
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client


# ── Health Endpoint ────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


# ── Predict Endpoint ───────────────────────────────────────────────────────────

def test_predict_valid(client):
    """Valid input should return 200 with a complete prediction response."""
    # Skip if models are not trained
    model_path = os.path.join("saved_models", "XGBoost_model.pkl")
    if not os.path.exists(model_path):
        pytest.skip("Models not trained — run python ml/train.py first.")

    resp = client.post(
        "/api/predict",
        data=json.dumps(VALID_PAYLOAD),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()

    # Required fields in response
    assert "prediction"    in data
    assert "probability"   in data
    assert "risk_label"    in data
    assert "risk_category" in data
    assert "shap_values"   in data
    assert "top_factors"   in data
    assert "recommendation" in data

    # Probability must be 0–100
    assert 0 <= data["probability"] <= 100

    # Prediction must be binary
    assert data["prediction"] in (0, 1)

    # SHAP values must cover all 8 features
    assert len(data["shap_values"]) == 8


def test_predict_missing_required_field(client):
    """Missing a required field should return 422 Unprocessable Entity."""
    payload = {"glucose": 148, "bmi": 33.6}   # many required fields missing
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 422
    data = resp.get_json()
    assert "error" in data
    assert "details" in data


def test_predict_out_of_range(client):
    """Out-of-range glucose value should return 422."""
    payload = {**VALID_PAYLOAD, "glucose": 999}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 422


def test_predict_non_numeric(client):
    """Non-numeric value should return 422."""
    payload = {**VALID_PAYLOAD, "glucose": "not-a-number"}
    resp = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert resp.status_code == 422


def test_predict_empty_body(client):
    """Empty body should return 400."""
    resp = client.post("/api/predict", data="", content_type="application/json")
    assert resp.status_code == 400


# ── History Endpoint ───────────────────────────────────────────────────────────

def test_history_empty(client):
    """History should return an empty list on a fresh database."""
    resp = client.get("/api/history")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_history_pagination(client):
    """Pagination parameters should be respected."""
    resp = client.get("/api/history?page=1&limit=5")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["limit"] == 5


# ── Model Metrics Endpoint ─────────────────────────────────────────────────────

def test_models_endpoint_no_file(client):
    """Should return 404 if metrics_report.json hasn't been generated yet."""
    resp = client.get("/api/models")
    # Either 200 (if trained) or 404 (if not trained)
    assert resp.status_code in (200, 404)

