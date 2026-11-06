# 🩺 Explainable Machine Learning-Based Diabetes Risk Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-XAI-purple)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A production-grade, full-stack web application that predicts diabetes risk using multiple ML algorithms and explains every prediction using SHAP-powered Explainable AI.**

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [ML Models & Performance](#-ml-models--performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [Disclaimer](#%EF%B8%8F-disclaimer)
- [License](#-license)

---

## 🎯 Project Overview

Early identification of elevated diabetes risk can support awareness and prompt appropriate medical consultation. This system addresses a critical limitation of traditional ML classifiers: **they predict without explaining**.

This capstone project delivers a **web-based, explainable ML system** that:

1. Collects relevant health parameters via an interactive form
2. Preprocesses and validates input data using a robust ML pipeline
3. Predicts diabetes risk using **5 ML algorithms** running in parallel
4. Returns a **probability score** and **risk category** (Low / Moderate / High)
5. Explains contributing factors using **SHAP** (global + local interpretability)
6. Stores every prediction in a **history dashboard** with timestamps
7. Provides a **Model Comparison Dashboard** with Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics

> **⚠️ Disclaimer:** This system is intended for educational and awareness purposes only. It is **not** a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Prediction** | Logistic Regression, Random Forest, XGBoost, SVM, KNN evaluated simultaneously |
| 🔍 **SHAP Explainability** | Waterfall plots, force plots, and feature importance charts per prediction |
| 📊 **Risk Categorization** | Low / Moderate / High risk labels with probability percentages |
| 🗂️ **Prediction History** | SQLite-backed history dashboard for all past predictions |
| 📈 **Model Comparison** | Side-by-side metric comparison across all 5 algorithms |
| 🛡️ **Input Validation** | Both client-side and server-side range/type validation |
| 🐳 **Dockerized** | One-command startup with Docker Compose |
| 📱 **Responsive UI** | Mobile-friendly React frontend with Tailwind CSS |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                │
│              React 18 + Tailwind CSS + Recharts                     │
│    [Input Form] → [Risk Dashboard] → [SHAP Charts] → [History]      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  HTTP/REST (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FLASK REST API (Backend)                           │
│   /api/predict  │  /api/history  │  /api/models  │  /api/health     │
│                                                                     │
│  ┌──────────────────┐    ┌────────────────────┐                     │
│  │  Preprocessing   │    │   SHAP Explainer   │                     │
│  │  Pipeline        │    │  (TreeExplainer /  │                     │
│  │  (Imputation +   │    │   LinearExplainer) │                     │
│  │   SMOTE + Scale) │    └────────────────────┘                     │
│  └──────────────────┘                                               │
│                                                                     │
│  ┌────────┐ ┌──────────┐ ┌─────────┐ ┌─────┐ ┌─────┐              │
│  │  LR    │ │   RFC    │ │ XGBoost │ │ SVM │ │ KNN │  ML Models   │
│  └────────┘ └──────────┘ └─────────┘ └─────┘ └─────┘              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  SQLAlchemy ORM
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SQLite Database                                │
│           predictions_history | model_metrics | users               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

> See [`TECH_STACK.md`](./TECH_STACK.md) for the full breakdown with version pinning and rationale.

| Layer | Technology |
|---|---|
| **ML & Data Science** | Python 3.11, Scikit-Learn, XGBoost, SHAP, Pandas, NumPy, imbalanced-learn |
| **Backend** | Flask 3, Flask-CORS, Flask-SQLAlchemy, Gunicorn |
| **Frontend** | React 18, Vite, Tailwind CSS, Recharts, Axios |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **DevOps** | Docker, Docker Compose, GitHub Actions |

---

## 📂 Dataset

The project uses the **Pima Indians Diabetes Dataset** (National Institute of Diabetes and Digestive and Kidney Diseases).

- **Bundled Dataset:** Included directly in this repository at [`backend/data/diabetes.csv`](./backend/data/diabetes.csv) (ready to train out-of-the-box).
- **Direct Raw CSV Mirror:** [GitHub Raw Mirror (Direct Download)](https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv)
- **Kaggle Benchmark:** [Kaggle Dataset Hub](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Secondary (optional lifestyle enrichment):** [CDC BRFSS Diabetes Health Indicators](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

### Input Features (Pima)

| Feature | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of pregnancies | Count |
| `Glucose` | Plasma glucose concentration (2h OGTT) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skinfold thickness | mm |
| `Insulin` | 2-hour serum insulin | μU/mL |
| `BMI` | Body Mass Index | kg/m² |
| `DiabetesPedigreeFunction` | Family history likelihood score | — |
| `Age` | Age | years |
| `Outcome` | **Target** — 0: No Diabetes, 1: Diabetes | Binary |

---

## 🤖 ML Models & Performance

> Metrics evaluated with **Stratified 10-Fold Cross-Validation** on preprocessed + SMOTE-balanced data.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~76% | ~72% | ~71% | ~71% | ~0.83 |
| Random Forest | ~80% | ~78% | ~74% | ~76% | ~0.88 |
| **XGBoost** | **~82%** | **~80%** | **~77%** | **~78%** | **~0.90** |
| SVM (RBF Kernel) | ~77% | ~74% | ~72% | ~73% | ~0.84 |
| K-Nearest Neighbors | ~74% | ~70% | ~68% | ~69% | ~0.80 |

> XGBoost is selected as the **primary prediction model** (best ROC-AUC). All models are shown on the comparison dashboard.

---

## 📁 Project Structure

```
capstone/
│
├── backend/                        # Flask API
│   ├── app.py                      # Application factory + routes
│   ├── config.py                   # Configuration (dev/prod)
│   ├── models/
│   │   └── db_models.py            # SQLAlchemy ORM models
│   ├── ml/
│   │   ├── train.py                # Model training + serialization
│   │   ├── predict.py              # Inference + SHAP explanation engine
│   │   ├── preprocess.py           # Data cleaning + feature engineering
│   │   └── evaluate.py             # Cross-validation + metrics computation
│   ├── routes/
│   │   ├── predict_routes.py       # POST /api/predict
│   │   ├── history_routes.py       # GET /api/history
│   │   └── model_routes.py         # GET /api/models
│   ├── saved_models/               # Serialized .pkl files (gitignored)
│   ├── data/
│   │   └── diabetes.csv            # Raw dataset
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # React + Vite app
│   ├── src/
│   │   ├── components/
│   │   │   ├── InputForm.jsx       # Health parameter input form
│   │   │   ├── RiskResult.jsx      # Risk score + category display
│   │   │   ├── ShapChart.jsx       # SHAP waterfall/bar chart
│   │   │   ├── ModelComparison.jsx # Metrics comparison table/chart
│   │   │   └── HistoryTable.jsx    # Past predictions list
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   └── History.jsx
│   │   ├── services/
│   │   │   └── api.js              # Axios API calls
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── notebooks/                      # Jupyter EDA + training notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_SHAP_Analysis.ipynb
│
├── docker-compose.yml
├── .env.example
├── README.md
├── TECH_STACK.md
└── DEVELOPMENT.md
```

---

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose **OR** Python 3.11+ and Node.js 20+
- Git

### Option 1: Docker (Recommended — One Command)

```bash
git clone https://github.com/your-username/diabetes-prediction.git
cd diabetes-prediction
cp .env.example .env
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:5000 |

---

### Option 2: Manual Setup

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train and serialize all ML models
python ml/train.py

# Start the Flask development server
flask run --port 5000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

### Train Models

```bash
cd backend
python ml/train.py
```

This script will:
1. Load and clean `data/diabetes.csv`
2. Apply median imputation for zero-encoded missing values
3. Split data (80/20 stratified)
4. Apply **SMOTE** on the training set only
5. Train 5 ML models with hyperparameter tuning (GridSearchCV)
6. Serialize all models + SHAP explainers to `saved_models/`
7. Output a `metrics_report.json` used by the comparison dashboard

---

## 📡 API Reference

### `POST /api/predict`
Submit health parameters and receive risk prediction + SHAP explanation.

**Request Body:**
```json
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
```

**Response:**
```json
{
  "prediction": 1,
  "risk_label": "High Risk",
  "probability": 82.4,
  "risk_category": "high",
  "shap_values": {
    "Glucose": 0.42,
    "BMI": 0.21,
    "Age": 0.15,
    "Pregnancies": 0.08,
    "DiabetesPedigreeFunction": 0.06,
    "BloodPressure": -0.04,
    "Insulin": -0.02,
    "SkinThickness": -0.01
  },
  "top_factors": ["Glucose", "BMI", "Age"],
  "recommendation": "Your glucose and BMI values are significantly elevated. Please consult a healthcare professional."
}
```

### `GET /api/history?page=1&limit=10`
Retrieve paginated prediction history.

### `GET /api/models`
Retrieve trained model evaluation metrics.

### `GET /api/health`
Health check endpoint.

---

## 🖼️ Screenshots

> *(Add screenshots of the input form, risk result dashboard, SHAP chart, and model comparison page here after UI is built)*

---

## ⚠️ Disclaimer

This application is a **capstone academic project** built for educational purposes. The predictions generated are **not medical diagnoses**. All results should be treated as informational awareness only. For any health concerns, please consult a licensed medical professional.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

<div align="center">
Built with ❤️ as an academic capstone project | Explainable AI for Healthcare
</div>

