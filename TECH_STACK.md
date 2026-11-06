# 🛠️ Tech Stack — Explainable ML Diabetes Risk Prediction System

> This document is the definitive reference for every technology, library, and tool used in this project — with version pinning, rationale, and alternatives considered.

---

## 📐 Architecture Overview

The system follows a **decoupled three-tier architecture**:

```
[ React Frontend (SPA) ]  ←HTTP/REST→  [ Flask Backend (API) ]  ←ORM→  [ SQLite / PostgreSQL ]
                                              ↑
                                   [ ML Pipeline + SHAP Engine ]
```

This separation means:
- Frontend and backend can be independently scaled and deployed
- ML models are served via a clean REST API — swappable without touching the UI
- Database schema is managed via SQLAlchemy ORM — easily switched from SQLite (dev) to PostgreSQL (prod)

---

## 1. 🐍 Machine Learning & Data Science Layer

### Python 3.11+
- **Why:** Python is the industry standard for ML. Version 3.11 offers significant performance improvements (~25% faster than 3.9).
- **Alternative considered:** Python 3.10 (also acceptable)

---

### Pandas 2.x + NumPy 1.26+
| Library | Role |
|---|---|
| `pandas` | DataFrame manipulation — loading CSV, cleaning, feature engineering |
| `numpy` | Numerical arrays, mathematical operations on feature vectors |

```python
# Example: Replacing zero-encoded missing values in Pima dataset
import pandas as pd
import numpy as np

df = pd.read_csv('data/diabetes.csv')
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
```

---

### Scikit-Learn 1.4+
**Core ML framework** providing:

| Component | Usage |
|---|---|
| `train_test_split` | Stratified 80/20 split |
| `StandardScaler` | Feature normalization |
| `SimpleImputer` | Median imputation for missing values |
| `Pipeline` | Chain preprocessing + model into a single serializable object |
| `GridSearchCV` | Hyperparameter tuning with cross-validation |
| `LogisticRegression` | Baseline interpretable model |
| `RandomForestClassifier` | Ensemble tree model |
| `SVC` | Support Vector Classifier (RBF kernel) |
| `KNeighborsClassifier` | Instance-based learning model |
| `classification_report` | Precision, Recall, F1-Score per class |
| `roc_auc_score` | ROC-AUC evaluation metric |

---

### XGBoost 2.x
- **Why:** Consistently top-performing gradient boosting library. Natively handles missing values, supports `scale_pos_weight` for class imbalance, and has native SHAP integration via `TreeExplainer`.
- **Alternative considered:** LightGBM (also excellent — faster on large data, slightly lower interpretability tooling)

---

### imbalanced-learn 0.12+ (`imblearn`)
- **Why:** The Pima dataset has ~65% negative / ~35% positive class split — a moderate imbalance that skews models toward predicting "no diabetes."
- **Usage:** `SMOTE` (Synthetic Minority Over-sampling Technique) is applied **only to the training set** after the train/test split to prevent data leakage.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

---

### SHAP 0.45+ (SHapley Additive exPlanations)
**The explainability core of the entire system.**

| Explainer Type | Used For |
|---|---|
| `shap.TreeExplainer` | Random Forest, XGBoost (tree-based models — fast and exact) |
| `shap.LinearExplainer` | Logistic Regression (linear models) |
| `shap.KernelExplainer` | SVM, KNN (model-agnostic — slower, used offline) |

**Outputs per prediction:**
- `shap_values`: Array of per-feature contributions for the specific patient
- Global summary: Feature importance across training dataset (pre-computed at training time)
- Waterfall chart data: Sent as JSON to React for visualization

**Why SHAP over LIME?**
- SHAP has mathematical guarantees (consistency, efficiency, symmetry) rooted in cooperative game theory (Shapley values)
- SHAP provides both **local** (per-patient) and **global** (dataset-wide) interpretations
- Native integration with tree-based models makes it fast enough for real-time web API responses
- LIME is model-agnostic but perturbation-based — less stable across repeated calls for the same input

---

### Joblib 1.3+
- Serializes trained ML Pipeline objects and SHAP Explainers to `.pkl` files
- More efficient than `pickle` for NumPy arrays (uses memory-mapped files)

---

### Matplotlib 3.8+ / Seaborn 0.13+
- Used **offline** in Jupyter notebooks for EDA visualizations, correlation heatmaps, and class distribution plots
- Not used at runtime in the Flask API (SHAP values are sent as JSON; charts are rendered by React/Recharts)

---

## 2. ⚙️ Backend Layer

### Flask 3.x
- **Why:** Lightweight Python micro-framework. No unnecessary boilerplate — perfect for serving ML model inference endpoints as a REST API.
- **Alternative considered:** FastAPI (excellent choice too — faster, async, auto OpenAPI docs; Flask chosen for wider academic familiarity and simpler debugging)

| Flask Extension | Role |
|---|---|
| `Flask-CORS` | Enables cross-origin requests from the React frontend on a different port |
| `Flask-SQLAlchemy` | ORM for interacting with SQLite/PostgreSQL without raw SQL |
| `Flask-Marshmallow` | Request/response schema validation and serialization |
| `python-dotenv` | Loads environment variables from `.env` file |

---

### Gunicorn 21+
- **Why:** Production-grade WSGI HTTP server. Flask's built-in server is single-threaded and not safe for production.
- Run command: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- 4 worker processes to handle concurrent prediction requests

---

### SQLAlchemy 2.x (ORM)
Manages two primary tables:

**`PredictionHistory` table:**
```
id | timestamp | pregnancies | glucose | bmi | ... | prediction | probability | risk_label | shap_json
```

**`ModelMetrics` table:**
```
id | model_name | accuracy | precision | recall | f1_score | roc_auc | trained_at
```

---

### Database: SQLite (Development) / PostgreSQL (Production)
| Environment | Database | Reason |
|---|---|---|
| Development | SQLite | Zero configuration, file-based, perfect for local dev and demos |
| Production | PostgreSQL 16 | ACID-compliant, concurrent writes, production-ready |

Switch is controlled by the `DATABASE_URL` environment variable in `.env`.

---

## 3. 🎨 Frontend Layer

### React 18 + Vite 5
- **Why React:** Component-based architecture makes it easy to build the multi-panel dashboard (form + result + SHAP chart + history)
- **Why Vite over Create React App:** 10-100x faster HMR (Hot Module Replacement) during development; modern ES module-based build
- **Alternative considered:** Next.js (adds SSR — overkill for this SPA; Vite is simpler)

---

### Tailwind CSS 3.4
- Utility-first CSS framework — no custom CSS files needed for most components
- Responsive grid system handles mobile layout automatically
- Color-coded risk output: green (Low), yellow (Moderate), red (High)

---

### Recharts 2.x
- React-native charting library
- Used for:
  - **Horizontal bar chart**: SHAP feature contributions (positive = red = risk-increasing, negative = green = risk-decreasing)
  - **Radar chart**: Model comparison across metrics
  - **Line chart**: Historical risk trend for a session

---

### Axios 1.x
- Promise-based HTTP client for all API calls to the Flask backend
- Configured with base URL and request/response interceptors for error handling

---

### React Router 6
- Client-side routing for three main pages:
  - `/` — Home / Input Form
  - `/result` — Prediction Result + SHAP Dashboard
  - `/history` — Prediction History
  - `/models` — Model Comparison Dashboard

---

## 4. 🗃️ Data & Storage Layer

| Storage | Used For |
|---|---|
| `data/diabetes.csv` | Raw Pima Indians dataset (committed to repo) |
| `saved_models/*.pkl` | Serialized model Pipelines + SHAP Explainers (gitignored, generated by `train.py`) |
| `SQLite / PostgreSQL` | Prediction history + model metrics |

---

## 5. 🐳 DevOps & Infrastructure

### Docker + Docker Compose
Two-service stack:

```yaml
# docker-compose.yml (simplified)
services:
  backend:
    build: ./backend
    ports: ["5000:5000"]
    volumes: ["./backend/data:/app/data"]
    environment:
      - DATABASE_URL=sqlite:///predictions.db
      - FLASK_ENV=development

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [backend]
```

---

### GitHub Actions (CI/CD)
- **On push to `main`:** Run Python unit tests (`pytest`) and linting (`ruff`)
- **On PR:** Run frontend lint (`eslint`) and type checks
- **Optional:** Auto-deploy to Render / Railway on merge

---

## 6. 🧪 Testing & Code Quality

| Tool | Purpose |
|---|---|
| `pytest` | Backend unit tests (prediction logic, preprocessing, API routes) |
| `pytest-cov` | Code coverage reporting |
| `ruff` | Fast Python linter (replaces flake8 + isort + pyupgrade) |
| `eslint` | JavaScript/React linting |
| `Prettier` | Frontend code formatting |

---

## 7. 📓 Development Tools

| Tool | Purpose |
|---|---|
| Jupyter Notebook | EDA, model training experiments, SHAP visualizations |
| VS Code | Primary IDE (recommended extensions: Pylance, ESLint, Tailwind IntelliSense) |
| Postman / Thunder Client | Manual API testing |
| Git + GitHub | Version control |

---

## 📦 Full Dependency List

### Backend (`backend/requirements.txt`)
```
flask==3.0.3
flask-cors==4.0.0
flask-sqlalchemy==3.1.1
flask-marshmallow==1.2.1
marshmallow-sqlalchemy==1.1.0
python-dotenv==1.0.1
gunicorn==22.0.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
imbalanced-learn==0.12.3
shap==0.45.1
joblib==1.4.2
matplotlib==3.8.4
seaborn==0.13.2
pytest==8.2.2
pytest-cov==5.0.0
ruff==0.4.8
```

### Frontend (`frontend/package.json` key deps)
```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.23.1",
    "axios": "^1.7.2",
    "recharts": "^2.12.7",
    "tailwindcss": "^3.4.4",
    "@headlessui/react": "^2.1.1"
  },
  "devDependencies": {
    "vite": "^5.3.1",
    "@vitejs/plugin-react": "^4.3.1",
    "eslint": "^9.6.0",
    "prettier": "^3.3.2"
  }
}
```

---

## 🔄 Technology Decision Matrix

| Decision | Chosen | Alternative | Reason for Choice |
|---|---|---|---|
| ML Framework | Scikit-Learn + XGBoost | TensorFlow / PyTorch | Tabular data doesn't need deep learning; sklearn pipelines are clean and serializable |
| Explainability | SHAP | LIME | SHAP is mathematically consistent, faster for tree models, provides global+local views |
| Backend Framework | Flask | FastAPI | Simpler for academic context; Flask widely documented for ML serving |
| Frontend Framework | React + Vite | Vue.js / Svelte | React has the largest ecosystem; easier to hire/collaborate |
| Database | SQLite → PostgreSQL | MongoDB | Structured prediction records fit a relational model well |
| Styling | Tailwind CSS | Material UI / Ant Design | Tailwind gives full control without opinionated component design |
| Containerization | Docker Compose | Kubernetes | Docker Compose is sufficient for a 2-service app at this scale |
| Class Imbalance | SMOTE | Class weights | SMOTE creates synthetic samples; more effective than weighting alone for this dataset size |

