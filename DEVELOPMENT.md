# 📐 Development Guide — Explainable ML Diabetes Risk Prediction System

> This is the complete, end-to-end engineering blueprint for building this project. Follow the phases in order. Each phase has clear deliverables, tasks, and implementation notes.

---

## Table of Contents

- [Development Principles](#-development-principles)
- [Phase 0: Project Setup & Environment](#phase-0-project-setup--environment)
- [Phase 1: Data Acquisition & EDA](#phase-1-data-acquisition--exploratory-data-analysis-eda)
- [Phase 2: Data Preprocessing Pipeline](#phase-2-data-preprocessing-pipeline)
- [Phase 3: Model Training & Evaluation](#phase-3-model-training--evaluation)
- [Phase 4: SHAP Explainability Integration](#phase-4-shap-explainability-integration)
- [Phase 5: Flask Backend API](#phase-5-flask-backend-api)
- [Phase 6: React Frontend](#phase-6-react-frontend)
- [Phase 7: Database Integration (Prediction History)](#phase-7-database-integration--prediction-history)
- [Phase 8: Dockerization](#phase-8-dockerization)
- [Phase 9: Testing & Quality Assurance](#phase-9-testing--quality-assurance)
- [Phase 10: Documentation & Final Submission](#phase-10-documentation--final-submission)
- [Coding Standards](#-coding-standards)
- [Git Workflow](#-git-workflow)
- [Environment Variables](#-environment-variables)
- [Known Pitfalls & Solutions](#-known-pitfalls--solutions)

---

## 🧭 Development Principles

1. **Never leak data.** SMOTE and scaling must only be fit on the **training set**, never the full dataset.
2. **Explain every prediction.** Every inference call must return SHAP values alongside the prediction.
3. **Validate at every boundary.** Validate inputs on the React form AND in the Flask route — never trust data from the client.
4. **Keep ML and API concerns separated.** The Flask route should not contain ML logic. It calls functions in `ml/predict.py`.
5. **Fail gracefully.** The system must return meaningful error messages — never a raw Python traceback — to the frontend.
6. **Reproducibility.** All random states, seeds, and hyperparameters must be documented and version-controlled.

---

## Phase 0: Project Setup & Environment

### 0.1 Repository Structure Initialization

```bash
mkdir capstone && cd capstone
mkdir -p backend/{ml,models,routes,data,saved_models,tests}
mkdir -p frontend/src/{components,pages,services}
mkdir -p notebooks
touch docker-compose.yml .env.example README.md TECH_STACK.md DEVELOPMENT.md
```

### 0.2 Backend Python Environment

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install flask flask-cors flask-sqlalchemy flask-marshmallow \
            marshmallow-sqlalchemy python-dotenv gunicorn \
            pandas numpy scikit-learn xgboost imbalanced-learn \
            shap joblib matplotlib seaborn jupyter \
            pytest pytest-cov ruff
pip freeze > requirements.txt
```

### 0.3 Frontend Node Environment

```bash
cd frontend
npm create vite@latest . -- --template react
npm install
npm install axios react-router-dom recharts @headlessui/react
npm install -D tailwindcss postcss autoprefixer eslint prettier
npx tailwindcss init -p
```

**Configure Tailwind** (`tailwind.config.js`):
```js
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

Add to `src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### 0.4 Environment Variables

Create `.env` in `backend/`:
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=sqlite:///predictions.db
CORS_ORIGIN=http://localhost:3000
MODEL_DIR=saved_models/
```

Create `.env.example` (committed to repo — no secrets):
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=change-me
DATABASE_URL=sqlite:///predictions.db
CORS_ORIGIN=http://localhost:3000
MODEL_DIR=saved_models/
```

---

## Phase 1: Data Acquisition & Exploratory Data Analysis (EDA)

### 1.1 Dataset Setup

The dataset is already bundled and pre-loaded at `backend/data/diabetes.csv`.

If you ever need to re-download or mirror the **Pima Indians Diabetes Dataset**:
- **Direct Raw Download:** `https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv`
- **Kaggle Page:** `https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database`

Place or verify the CSV file at `backend/data/diabetes.csv`.

**Columns:** `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

### 1.2 EDA Notebook (`notebooks/01_EDA.ipynb`)

Perform and document the following analyses:

#### 1.2.1 Basic Profiling
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../backend/data/diabetes.csv')
print(df.shape)         # (768, 9)
print(df.dtypes)
print(df.describe())
print(df['Outcome'].value_counts())  # Class distribution
```

#### 1.2.2 Class Imbalance Analysis
```python
# Expected: ~500 (class 0) vs ~268 (class 1) — 65/35 split
df['Outcome'].value_counts(normalize=True).plot(kind='bar', title='Class Distribution')
```

#### 1.2.3 Zero-Value Investigation (Missing Data)
```python
# In this dataset, 0s in physiological columns represent MISSING values
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
(df[zero_cols] == 0).sum()
# Insulin has ~374 zeros — the most problematic feature
```

#### 1.2.4 Correlation Heatmap
```python
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
# Glucose, BMI, Age are expected to be most correlated with Outcome
```

#### 1.2.5 Feature Distributions by Class
```python
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
features = df.columns[:-1]
for ax, feat in zip(axes.flat, features):
    df.groupby('Outcome')[feat].plot(kind='kde', ax=ax, legend=True)
    ax.set_title(feat)
plt.tight_layout()
```

#### 1.2.6 Outlier Detection
```python
# Use IQR method per feature; document but don't aggressively remove
# (clinical data outliers may be real and significant)
for col in df.columns[:-1]:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers")
```

**EDA Deliverables:**
- [ ] Class distribution plot
- [ ] Zero-value count table per column
- [ ] Correlation heatmap
- [ ] Feature distribution KDE plots by outcome
- [ ] Written summary of key findings (3–5 bullet points per feature)

---

## Phase 2: Data Preprocessing Pipeline

### 2.1 Preprocessing Module (`backend/ml/preprocess.py`)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os

ZERO_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
TARGET = 'Outcome'


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load dataset and replace physiological zero-values with NaN."""
    df = pd.read_csv(filepath)
    df[ZERO_COLS] = df[ZERO_COLS].replace(0, np.nan)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split preserving class ratio."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


def build_preprocessor() -> Pipeline:
    """
    Scikit-Learn pipeline: median imputation → standard scaling.
    This pipeline is fit only on training data to prevent leakage.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to training data only.
    SMOTE creates synthetic minority class samples to balance the dataset.
    CRITICAL: Never apply SMOTE to test/validation data.
    """
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)


def prepare_data(filepath: str):
    """Full preprocessing pipeline. Returns processed splits."""
    df = load_and_clean(filepath)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)   # fit + transform on train
    X_test_scaled = preprocessor.transform(X_test)          # only transform on test

    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res, y_test, preprocessor
```

> **⚠️ Critical:** `preprocessor.fit_transform()` is called only on `X_train`. `X_test` only gets `.transform()`. This is the most common source of data leakage in student projects.

### 2.2 Preprocessing Notebook (`notebooks/02_Preprocessing.ipynb`)
- Visualize before/after imputation distributions
- Show class balance before and after SMOTE
- Verify no NaN values remain after pipeline

---

## Phase 3: Model Training & Evaluation

### 3.1 Model Definitions (`backend/ml/train.py`)

```python
import joblib
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)
from xgboost import XGBClassifier
import shap
from preprocess import prepare_data, FEATURES

DATA_PATH = 'data/diabetes.csv'
MODEL_DIR = 'saved_models/'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Model Definitions ──────────────────────────────────────────────────────────
MODELS = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss',
                                use_label_encoder=False),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall': round(recall_score(y_test, y_pred), 4),
        'f1_score': round(f1_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4)
    }


def train_all_models():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(DATA_PATH)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_metrics = {}

    for name, config in MODELS.items():
        print(f"\n[Training] {name}...")
        grid = GridSearchCV(config['model'], config['params'],
                            cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics = evaluate_model(best_model, X_test, y_test)
        metrics['best_params'] = grid.best_params_
        all_metrics[name] = metrics
        print(f"  Best Params: {grid.best_params_}")
        print(f"  Test Metrics: {metrics}")

        # Save model
        joblib.dump(best_model, f'{MODEL_DIR}{name}_model.pkl')

        # Build and save SHAP Explainer
        if name in ('RandomForest', 'XGBoost'):
            explainer = shap.TreeExplainer(best_model)
        elif name == 'LogisticRegression':
            explainer = shap.LinearExplainer(best_model, X_train)
        else:
            # KernelExplainer is slow — use a small background dataset
            background = shap.sample(X_train, 100)
            explainer = shap.KernelExplainer(best_model.predict_proba, background)
        joblib.dump(explainer, f'{MODEL_DIR}{name}_explainer.pkl')

    # Save preprocessor
    joblib.dump(preprocessor, f'{MODEL_DIR}preprocessor.pkl')

    # Save metrics report
    with open(f'{MODEL_DIR}metrics_report.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("\n✅ All models trained and serialized successfully.")
    return all_metrics


if __name__ == '__main__':
    train_all_models()
```

### 3.2 Evaluation Notebook (`notebooks/03_Model_Training.ipynb`)
Visualizations to include:
- **Confusion matrices** for each model (2×2 heatmap)
- **ROC curves** for all 5 models on a single plot
- **Precision-Recall curves**
- **Learning curves** (train vs. validation score vs. training set size) for XGBoost and Random Forest
- **Bar chart comparison** of all metrics across models

---

## Phase 4: SHAP Explainability Integration

### 4.1 Prediction + Explanation Engine (`backend/ml/predict.py`)

```python
import joblib
import numpy as np
import pandas as pd
import shap
import os

MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models/')
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
PRIMARY_MODEL = 'XGBoost'


def load_artifacts():
    """Load model, explainer, and preprocessor from disk."""
    model = joblib.load(f'{MODEL_DIR}{PRIMARY_MODEL}_model.pkl')
    explainer = joblib.load(f'{MODEL_DIR}{PRIMARY_MODEL}_explainer.pkl')
    preprocessor = joblib.load(f'{MODEL_DIR}preprocessor.pkl')
    return model, explainer, preprocessor


def get_risk_label(probability: float) -> dict:
    """Convert probability to risk category."""
    if probability < 0.35:
        return {'label': 'Low Risk', 'category': 'low', 'color': 'green'}
    elif probability < 0.65:
        return {'label': 'Moderate Risk', 'category': 'moderate', 'color': 'yellow'}
    else:
        return {'label': 'High Risk', 'category': 'high', 'color': 'red'}


def generate_recommendation(shap_values: dict, risk_category: str) -> str:
    """Generate a plain-language recommendation based on top SHAP features."""
    top_risk_factors = [k for k, v in sorted(shap_values.items(),
                                               key=lambda x: x[1], reverse=True)
                        if v > 0][:3]
    factor_text = ', '.join(top_risk_factors) if top_risk_factors else 'your health metrics'
    if risk_category == 'high':
        return (f"Your prediction indicates HIGH risk, primarily driven by {factor_text}. "
                f"Please consult a healthcare professional as soon as possible for a formal evaluation.")
    elif risk_category == 'moderate':
        return (f"Your prediction indicates MODERATE risk. Key factors include {factor_text}. "
                f"Consider scheduling a check-up with your doctor and reviewing your lifestyle choices.")
    else:
        return ("Your prediction indicates LOW risk. Maintain a healthy lifestyle with "
                "regular exercise, balanced diet, and routine health check-ups.")


def predict(input_data: dict) -> dict:
    """
    Full inference pipeline:
    1. Validate and structure input
    2. Preprocess (impute + scale)
    3. Predict probability
    4. Compute SHAP values
    5. Return structured response
    """
    model, explainer, preprocessor = load_artifacts()

    # Structure input as DataFrame (preserves feature names for SHAP)
    input_df = pd.DataFrame([{k: float(v) for k, v in input_data.items()}],
                             columns=FEATURES)

    # Preprocess (apply same transforms as training)
    input_scaled = preprocessor.transform(input_df)

    # Predict
    probability = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(probability >= 0.5)
    risk_info = get_risk_label(probability)

    # SHAP explanation
    shap_vals = explainer.shap_values(input_scaled)
    # For binary classifiers: shap_values returns [class0_vals, class1_vals]
    if isinstance(shap_vals, list):
        shap_for_positive_class = shap_vals[1][0]
    else:
        shap_for_positive_class = shap_vals[0]

    shap_dict = {feat: round(float(val), 4)
                 for feat, val in zip(FEATURES, shap_for_positive_class)}

    recommendation = generate_recommendation(shap_dict, risk_info['category'])

    return {
        'prediction': prediction,
        'probability': round(probability * 100, 2),
        'risk_label': risk_info['label'],
        'risk_category': risk_info['category'],
        'risk_color': risk_info['color'],
        'shap_values': shap_dict,
        'top_factors': sorted(shap_dict, key=shap_dict.get, reverse=True)[:3],
        'recommendation': recommendation
    }
```

### 4.2 SHAP Analysis Notebook (`notebooks/04_SHAP_Analysis.ipynb`)
- **Summary plot:** `shap.summary_plot(shap_values, X_test, plot_type='bar')`
- **Beeswarm plot:** `shap.summary_plot(shap_values, X_test)` — shows distribution of impact
- **Waterfall plot for single patient:** `shap.waterfall_plot(shap.Explanation(...))`
- **Force plot:** `shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])`
- **Dependence plot for top features:** `shap.dependence_plot('Glucose', shap_values[1], X_test)`

---

## Phase 5: Flask Backend API

### 5.1 App Factory (`backend/app.py`)

```python
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

load_dotenv()
db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///predictions.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret')

    CORS(app, resources={r"/api/*": {"origins": os.getenv('CORS_ORIGIN', '*')}})
    db.init_app(app)

    from routes.predict_routes import predict_bp
    from routes.history_routes import history_bp
    from routes.model_routes import model_bp

    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(history_bp, url_prefix='/api')
    app.register_blueprint(model_bp, url_prefix='/api')

    @app.route('/api/health')
    def health():
        return {'status': 'ok', 'service': 'diabetes-prediction-api'}

    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
```

### 5.2 Database Models (`backend/models/db_models.py`)

```python
from app import db
from datetime import datetime
import json


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Input features
    pregnancies = db.Column(db.Float)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree = db.Column(db.Float)
    age = db.Column(db.Float)

    # Output
    prediction = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    risk_label = db.Column(db.String(20), nullable=False)
    risk_category = db.Column(db.String(10), nullable=False)
    shap_json = db.Column(db.Text)  # Store SHAP values as JSON string

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'inputs': {
                'pregnancies': self.pregnancies,
                'glucose': self.glucose,
                'bloodPressure': self.blood_pressure,
                'skinThickness': self.skin_thickness,
                'insulin': self.insulin,
                'bmi': self.bmi,
                'diabetesPedigreeFunction': self.diabetes_pedigree,
                'age': self.age
            },
            'prediction': self.prediction,
            'probability': self.probability,
            'risk_label': self.risk_label,
            'risk_category': self.risk_category,
            'shap_values': json.loads(self.shap_json) if self.shap_json else {}
        }
```

### 5.3 Prediction Route (`backend/routes/predict_routes.py`)

```python
from flask import Blueprint, request, jsonify
from models.db_models import PredictionHistory
from app import db
from ml.predict import predict
import json

predict_bp = Blueprint('predict', __name__)

# Input validation ranges (clinically reasonable bounds)
VALIDATION_RULES = {
    'pregnancies':              {'min': 0, 'max': 20, 'required': True},
    'glucose':                  {'min': 50, 'max': 300, 'required': True},
    'bloodPressure':            {'min': 30, 'max': 180, 'required': True},
    'skinThickness':            {'min': 0, 'max': 100, 'required': False},
    'insulin':                  {'min': 0, 'max': 900, 'required': False},
    'bmi':                      {'min': 10.0, 'max': 80.0, 'required': True},
    'diabetesPedigreeFunction': {'min': 0.0, 'max': 3.0, 'required': True},
    'age':                      {'min': 1, 'max': 120, 'required': True}
}


def validate_input(data: dict) -> list:
    """Returns a list of validation error strings, empty if all valid."""
    errors = []
    for field, rules in VALIDATION_RULES.items():
        value = data.get(field)
        if rules['required'] and value is None:
            errors.append(f"'{field}' is required.")
            continue
        if value is not None:
            try:
                val = float(value)
                if val < rules['min'] or val > rules['max']:
                    errors.append(f"'{field}' must be between {rules['min']} and {rules['max']}.")
            except (TypeError, ValueError):
                errors.append(f"'{field}' must be a numeric value.")
    return errors


@predict_bp.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON.'}), 400

    errors = validate_input(data)
    if errors:
        return jsonify({'error': 'Validation failed.', 'details': errors}), 422

    try:
        # Rename keys to match FEATURES list
        ml_input = {
            'Pregnancies': data['pregnancies'],
            'Glucose': data['glucose'],
            'BloodPressure': data['bloodPressure'],
            'SkinThickness': data.get('skinThickness', 0),
            'Insulin': data.get('insulin', 0),
            'BMI': data['bmi'],
            'DiabetesPedigreeFunction': data['diabetesPedigreeFunction'],
            'Age': data['age']
        }

        result = predict(ml_input)

        # Save to database
        record = PredictionHistory(
            pregnancies=data['pregnancies'],
            glucose=data['glucose'],
            blood_pressure=data['bloodPressure'],
            skin_thickness=data.get('skinThickness', 0),
            insulin=data.get('insulin', 0),
            bmi=data['bmi'],
            diabetes_pedigree=data['diabetesPedigreeFunction'],
            age=data['age'],
            prediction=result['prediction'],
            probability=result['probability'],
            risk_label=result['risk_label'],
            risk_category=result['risk_category'],
            shap_json=json.dumps(result['shap_values'])
        )
        db.session.add(record)
        db.session.commit()

        result['record_id'] = record.id
        return jsonify(result), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Prediction failed.', 'message': str(e)}), 500
```

### 5.4 History Route (`backend/routes/history_routes.py`)

```python
from flask import Blueprint, request, jsonify
from models.db_models import PredictionHistory

history_bp = Blueprint('history', __name__)


@history_bp.route('/history', methods=['GET'])
def get_history():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    limit = min(limit, 100)  # Cap at 100 per page

    pagination = (PredictionHistory.query
                  .order_by(PredictionHistory.timestamp.desc())
                  .paginate(page=page, per_page=limit, error_out=False))

    return jsonify({
        'predictions': [p.to_dict() for p in pagination.items],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages
    }), 200


@history_bp.route('/history/<int:record_id>', methods=['DELETE'])
def delete_prediction(record_id):
    record = PredictionHistory.query.get_or_404(record_id)
    from app import db
    db.session.delete(record)
    db.session.commit()
    return jsonify({'message': f'Record {record_id} deleted.'}), 200
```

### 5.5 Model Metrics Route (`backend/routes/model_routes.py`)

```python
from flask import Blueprint, jsonify
import json
import os

model_bp = Blueprint('models', __name__)
MODEL_DIR = os.getenv('MODEL_DIR', 'saved_models/')


@model_bp.route('/models', methods=['GET'])
def get_model_metrics():
    metrics_path = os.path.join(MODEL_DIR, 'metrics_report.json')
    if not os.path.exists(metrics_path):
        return jsonify({'error': 'Models not trained yet. Run python ml/train.py'}), 404
    with open(metrics_path) as f:
        metrics = json.load(f)
    return jsonify(metrics), 200
```

---

## Phase 6: React Frontend

### 6.1 Pages & Routing (`frontend/src/App.jsx`)

```jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import History from './pages/History';
import Models from './pages/Models';
import Navbar from './components/Navbar';

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-slate-50">
        <Navbar />
        <main className="max-w-5xl mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/result" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
```

### 6.2 Key Component Specifications

#### `InputForm.jsx`
- 8 labeled input fields with:
  - Placeholder values (typical healthy ranges shown as hints)
  - Min/max HTML5 validation
  - Tooltip explaining what each feature means (e.g., "Glucose: plasma glucose concentration from a 2-hour oral glucose tolerance test")
- Submit button triggers loading state
- On success: navigate to `/result` and pass result data via React Router state

#### `RiskResult.jsx`
- Large colored badge: 🟢 Low / 🟡 Moderate / 🔴 High
- Circular probability gauge (using Recharts `RadialBarChart`)
- Recommendation text box
- "View Detailed SHAP Explanation" expand/collapse section

#### `ShapChart.jsx`
- Horizontal bar chart (Recharts `BarChart`)
- Positive SHAP values = red bars ("Increases Risk")
- Negative SHAP values = green bars ("Decreases Risk")
- Feature labels on Y-axis, SHAP values on X-axis
- Reference line at 0

#### `ModelComparison.jsx`
- Table showing all 5 models × 5 metrics
- Highlight best value per metric column
- Radar chart for visual comparison

#### `HistoryTable.jsx`
- Paginated table of past predictions
- Columns: Date, Age, Glucose, BMI, Risk Level, Probability, View SHAP
- Color-coded Risk Level cell

### 6.3 API Service (`frontend/src/services/api.js`)

```javascript
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000
});

// Response interceptor for consistent error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.error || 'Network error. Please try again.';
    return Promise.reject(new Error(message));
  }
);

export const predictDiabetes = (formData) => api.post('/api/predict', formData);
export const getPredictionHistory = (page = 1, limit = 10) =>
  api.get(`/api/history?page=${page}&limit=${limit}`);
export const getModelMetrics = () => api.get('/api/models');
export const checkHealth = () => api.get('/api/health');
```

---

## Phase 7: Database Integration & Prediction History

### 7.1 Schema Design

The SQLite database contains two tables:

```sql
-- Automatically created by SQLAlchemy db.create_all()
CREATE TABLE prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pregnancies REAL,
    glucose REAL,
    blood_pressure REAL,
    skin_thickness REAL,
    insulin REAL,
    bmi REAL,
    diabetes_pedigree REAL,
    age REAL,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    risk_label VARCHAR(20) NOT NULL,
    risk_category VARCHAR(10) NOT NULL,
    shap_json TEXT
);
```

### 7.2 Database Migrations
For development, `db.create_all()` in the app factory is sufficient.
For production, use **Flask-Migrate** (Alembic wrapper):

```bash
pip install flask-migrate
flask db init
flask db migrate -m "initial schema"
flask db upgrade
```

---

## Phase 8: Dockerization

### 8.1 Backend Dockerfile (`backend/Dockerfile`)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models at build time (or mount pre-trained models as a volume)
RUN python ml/train.py

EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app()"]
```

### 8.2 Frontend Dockerfile (`frontend/Dockerfile`)

```dockerfile
# Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

**`frontend/nginx.conf`:**
```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;  # React Router SPA support
    }

    location /api/ {
        proxy_pass http://backend:5000;
        proxy_set_header Host $host;
    }
}
```

### 8.3 Docker Compose (`docker-compose.yml`)

```yaml
version: '3.9'

services:
  backend:
    build: ./backend
    container_name: diabetes_backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=sqlite:///predictions.db
      - CORS_ORIGIN=http://localhost:3000
      - MODEL_DIR=/app/saved_models/
    volumes:
      - ./backend/data:/app/data
      - model_cache:/app/saved_models
      - db_data:/app/instance
    restart: unless-stopped

  frontend:
    build: ./frontend
    container_name: diabetes_frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  model_cache:
  db_data:
```

---

## Phase 9: Testing & Quality Assurance

### 9.1 Backend Unit Tests (`backend/tests/`)

**`test_preprocessing.py`:**
```python
import pytest
import pandas as pd
import numpy as np
from ml.preprocess import load_and_clean, build_preprocessor, apply_smote

def test_zero_replacement():
    """Zero-encoded values in clinical columns should become NaN."""
    df = load_and_clean('data/diabetes.csv')
    for col in ['Glucose', 'BloodPressure', 'BMI']:
        assert df[col].min() > 0 or pd.isna(df[col]).any(), \
            f"{col} still has zero values after cleaning"

def test_no_data_leakage():
    """Preprocessor must be fit only on training data."""
    from ml.preprocess import prepare_data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data('data/diabetes.csv')
    # Verify shapes are consistent
    assert X_train.shape[1] == X_test.shape[1] == 8
```

**`test_predict_route.py`:**
```python
import pytest
import json
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.test_client() as client:
        with app.app_context():
            from app import db
            db.create_all()
        yield client

def test_predict_valid_input(client):
    payload = {
        "pregnancies": 2, "glucose": 148, "bloodPressure": 72,
        "skinThickness": 35, "insulin": 0, "bmi": 33.6,
        "diabetesPedigreeFunction": 0.627, "age": 50
    }
    response = client.post('/api/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'probability' in data
    assert 'shap_values' in data
    assert data['probability'] >= 0 and data['probability'] <= 100

def test_predict_missing_required_field(client):
    payload = {"glucose": 148, "bmi": 33.6}  # Missing required fields
    response = client.post('/api/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 422

def test_predict_out_of_range(client):
    payload = {
        "pregnancies": 2, "glucose": 999,  # Out of range
        "bloodPressure": 72, "skinThickness": 35,
        "insulin": 0, "bmi": 33.6,
        "diabetesPedigreeFunction": 0.627, "age": 50
    }
    response = client.post('/api/predict',
                           data=json.dumps(payload),
                           content_type='application/json')
    assert response.status_code == 422

def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
```

Run tests:
```bash
cd backend
pytest tests/ -v --cov=. --cov-report=term-missing
```

### 9.2 Frontend Linting

```bash
cd frontend
npx eslint src/ --ext .js,.jsx
npx prettier --check src/
```

### 9.3 Manual QA Checklist

- [ ] Submit form with all valid inputs → prediction result displayed correctly
- [ ] Submit form with blank required fields → validation errors shown
- [ ] Submit form with out-of-range values → validation errors shown
- [ ] SHAP chart renders with correct positive/negative coloring
- [ ] History page loads past predictions with correct timestamps
- [ ] Model comparison page displays all 5 models with metrics
- [ ] Application works on mobile viewport (375px width)
- [ ] API returns 422 for malformed JSON body
- [ ] Docker Compose brings up both services cleanly from cold start

---

## Phase 10: Documentation & Final Submission

### 10.1 Final Checklist

**Code:**
- [ ] All `TODO` comments resolved
- [ ] No hardcoded secrets or credentials in code
- [ ] `.gitignore` excludes `venv/`, `saved_models/`, `*.db`, `.env`, `__pycache__/`
- [ ] `requirements.txt` and `package.json` are up-to-date and pinned

**Documentation:**
- [ ] `README.md` has setup instructions, architecture diagram, API reference, screenshots
- [ ] `TECH_STACK.md` updated with final chosen versions
- [ ] All Jupyter notebooks have markdown cells explaining each step
- [ ] `DEVELOPMENT.md` updated if implementation deviated from this guide
- [ ] Model metrics table in README reflects actual trained model performance

**Presentation:**
- [ ] Architecture diagram (can use draw.io / Excalidraw)
- [ ] Demo recording (Loom or OBS — 5–10 min walkthrough)
- [ ] Slide deck with: Problem, Solution, Architecture, Dataset, Models, Results, SHAP demo, Limitations, Future Work

### 10.2 Suggested Future Enhancements

1. **HbA1c integration:** Add HbA1c (glycated hemoglobin) as an input feature — it is the gold-standard clinical diabetes marker
2. **User authentication:** Add JWT-based login so patients can view only their own history
3. **LLM-powered recommendations:** Pipe SHAP values to Gemini API to generate personalized, natural-language health advice
4. **Model retraining pipeline:** Admin endpoint that accepts new data and triggers incremental retraining
5. **Multi-dataset support:** Add CDC BRFSS dataset with a toggle for clinical vs. lifestyle-based prediction
6. **PDF report export:** Let users download a PDF of their prediction result + SHAP chart
7. **PWA:** Convert React app to a Progressive Web App for offline access

---

## 📏 Coding Standards

### Python
- Follow **PEP 8** (enforced by `ruff`)
- All functions must have docstrings explaining parameters and return values
- Use type hints for all function signatures
- Max line length: 100 characters

### JavaScript / React
- Use **ES6+** syntax (arrow functions, destructuring, optional chaining)
- Functional components only — no class components
- Component files are PascalCase (`InputForm.jsx`)
- Utility/service files are camelCase (`api.js`)
- Extract repeated logic into custom hooks (`usePredict.js`, `useHistory.js`)

---

## 🔀 Git Workflow

```bash
# Feature branch per phase
git checkout -b feature/phase-1-eda
git checkout -b feature/phase-2-preprocessing
git checkout -b feature/phase-3-training
git checkout -b feature/phase-4-shap
git checkout -b feature/phase-5-backend-api
git checkout -b feature/phase-6-frontend
git checkout -b feature/phase-7-database
git checkout -b feature/phase-8-docker

# Commit message format: <type>(<scope>): <short description>
# Examples:
git commit -m "feat(ml): add SHAP TreeExplainer for XGBoost"
git commit -m "fix(api): handle None insulin value in preprocessing"
git commit -m "docs(readme): add API response example"
git commit -m "test(backend): add predict route unit tests"
```

---

## 🌍 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_APP` | `app.py` | Flask application module |
| `FLASK_ENV` | `development` | `development` or `production` |
| `SECRET_KEY` | — | Flask session secret (required) |
| `DATABASE_URL` | `sqlite:///predictions.db` | SQLAlchemy database URI |
| `CORS_ORIGIN` | `http://localhost:3000` | Allowed CORS origin |
| `MODEL_DIR` | `saved_models/` | Path to serialized model files |
| `VITE_API_URL` | `http://localhost:5000` | Frontend API base URL (prefix with VITE_) |

---

## ⚠️ Known Pitfalls & Solutions

| Pitfall | Solution |
|---|---|
| **Data leakage via scaler** | Fit `StandardScaler` ONLY on `X_train`, then `.transform()` on `X_test` and inference inputs |
| **SMOTE before split** | Always split first, then apply SMOTE to `X_train` only |
| **SHAP `shap_values` list vs array** | For binary classifiers, `shap_values` is a list of 2 arrays — use `shap_values[1]` for positive class |
| **Zero-values as missing data** | The Pima dataset encodes missing values as 0 — replace with `NaN` before imputation |
| **Flask development server in production** | Use Gunicorn with multiple workers in Docker |
| **React Router 404 on page refresh** | Configure Nginx `try_files $uri /index.html` for SPA routing |
| **CORS errors** | Ensure Flask-CORS `origins` matches the exact frontend URL including port |
| **Joblib version mismatch** | Pin both `joblib` and `scikit-learn` versions — model files are not forward-compatible between major sklearn versions |
| **SHAP KernelExplainer is very slow** | Use `TreeExplainer` for RF/XGBoost; precompute `shap_values` for SVM/KNN at training time for the comparison dashboard |

