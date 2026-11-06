"""
ml/train.py
Train all 5 ML models, evaluate them, build SHAP explainers,
and serialize everything to disk.

Run this script once before starting the Flask server:
    cd backend && python ml/train.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)

from xgboost import XGBClassifier
import shap

# Allow imports from the backend/ root when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import prepare_data, FEATURES

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Model Registry ─────────────────────────────────────────────────────────────
# Each entry defines the model and its hyperparameter search space.
MODEL_REGISTRY = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {
            "C":      [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {
            "n_estimators":   [100, 200],
            "max_depth":      [None, 10, 20],
            "min_samples_split": [2, 5],
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        ),
        "params": {
            "n_estimators":  [100, 200],
            "max_depth":     [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample":     [0.8, 1.0],
        },
    },
    "SVM": {
        "model": SVC(probability=True, random_state=RANDOM_STATE),
        "params": {
            "C":      [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma":  ["scale", "auto"],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 11],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
    },
}


def evaluate_model(model, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """
    Compute all standard classification metrics for a trained model.

    Returns:
        Dictionary with accuracy, precision, recall, f1_score, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)),           4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)),             4),
    }


def build_shap_explainer(name: str, model, X_train: np.ndarray):
    """
    Build the appropriate SHAP explainer for a given model type.

    - TreeExplainer  → RandomForest, XGBoost (fast, exact)
    - LinearExplainer → LogisticRegression
    - KernelExplainer → SVM, KNN (model-agnostic, slower — use small background)
    """
    print(f"  [SHAP] Building explainer for {name}...")

    if name in ("RandomForest", "XGBoost"):
        explainer = shap.TreeExplainer(model)
    elif name == "LogisticRegression":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # KernelExplainer: use a 100-sample background dataset to keep inference fast
        background = shap.sample(X_train, 100, random_state=RANDOM_STATE)
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def train_all_models() -> dict:
    """
    Full training pipeline:
      1. Preprocess data
      2. For each model: GridSearchCV tuning → evaluate → serialize
      3. Build SHAP explainer → serialize
      4. Write metrics report to JSON

    Returns:
        Dictionary of model_name → metrics.
    """
    print("\n" + "=" * 60)
    print("  Diabetes Prediction — Model Training Pipeline")
    print("=" * 60)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(DATA_PATH)

    # 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    all_metrics = {}

    for name, config in MODEL_REGISTRY.items():
        print(f"\n[Training] {name}...")

        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics = evaluate_model(best_model, X_test, y_test)
        metrics["best_params"] = grid.best_params_
        metrics["cv_roc_auc"]  = round(float(grid.best_score_), 4)
        all_metrics[name] = metrics

        print(f"  Best params : {grid.best_params_}")
        print(f"  Test metrics: accuracy={metrics['accuracy']}  "
              f"f1={metrics['f1_score']}  roc_auc={metrics['roc_auc']}")
        print(classification_report(y_test, best_model.predict(X_test),
                                     target_names=["No Diabetes", "Diabetes"]))

        # Serialize model
        model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"  Saved model → {model_path}")

        # Build and serialize SHAP explainer
        explainer = build_shap_explainer(name, best_model, X_train)
        explainer_path = os.path.join(MODEL_DIR, f"{name}_explainer.pkl")
        joblib.dump(explainer, explainer_path)
        print(f"  Saved explainer → {explainer_path}")

    # Save preprocessor (used at inference time)
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n[Done] Saved preprocessor → {preprocessor_path}")

    # Save metrics report (read by /api/models endpoint)
    metrics_path = os.path.join(MODEL_DIR, "metrics_report.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[Done] Saved metrics report → {metrics_path}")

    print("\n✅ All models trained and serialized successfully!\n")
    return all_metrics


if __name__ == "__main__":
    train_all_models()

