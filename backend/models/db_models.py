"""
models/db_models.py
SQLAlchemy ORM models for the prediction history database.
"""

import json
from datetime import datetime
from app import db


class PredictionHistory(db.Model):
    """
    Stores every diabetes risk prediction made through the web app.
    Each record contains the input health parameters, the model output,
    and the SHAP explanation as a JSON string.
    """
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # ── Input Health Parameters ───────────────────────────────────────────────
    pregnancies             = db.Column(db.Float, nullable=False)
    glucose                 = db.Column(db.Float, nullable=False)
    blood_pressure          = db.Column(db.Float, nullable=False)
    skin_thickness          = db.Column(db.Float, nullable=True)
    insulin                 = db.Column(db.Float, nullable=True)
    bmi                     = db.Column(db.Float, nullable=False)
    diabetes_pedigree       = db.Column(db.Float, nullable=False)
    age                     = db.Column(db.Float, nullable=False)

    # ── Prediction Output ─────────────────────────────────────────────────────
    prediction              = db.Column(db.Integer, nullable=False)   # 0 or 1
    probability             = db.Column(db.Float, nullable=False)     # 0–100
    risk_label              = db.Column(db.String(20), nullable=False)
    risk_category           = db.Column(db.String(10), nullable=False) # low/moderate/high

    # ── Explainability ────────────────────────────────────────────────────────
    shap_json               = db.Column(db.Text, nullable=True)        # JSON string

    def to_dict(self) -> dict:
        """Serialize model instance to a JSON-safe dictionary."""
        return {
            "id":           self.id,
            "timestamp":    self.timestamp.isoformat(),
            "inputs": {
                "pregnancies":              self.pregnancies,
                "glucose":                  self.glucose,
                "bloodPressure":            self.blood_pressure,
                "skinThickness":            self.skin_thickness,
                "insulin":                  self.insulin,
                "bmi":                      self.bmi,
                "diabetesPedigreeFunction": self.diabetes_pedigree,
                "age":                      self.age,
            },
            "prediction":       self.prediction,
            "probability":      self.probability,
            "risk_label":       self.risk_label,
            "risk_category":    self.risk_category,
            "shap_values":      json.loads(self.shap_json) if self.shap_json else {},
        }

    def __repr__(self):
        return f"<Prediction id={self.id} risk={self.risk_label} prob={self.probability}%>"

