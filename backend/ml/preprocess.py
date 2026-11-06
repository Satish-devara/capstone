"""
ml/preprocess.py
Data loading, cleaning, and preprocessing pipeline.

IMPORTANT — Data Leakage Prevention:
  - StandardScaler is fit ONLY on X_train, then .transform() on X_test.
  - SMOTE is applied ONLY to training data after the split.
  - Never call .fit() or .fit_transform() with test data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ── Constants ──────────────────────────────────────────────────────────────────
# In the Pima dataset, these physiological columns encode missing values as 0.
# A glucose of 0 or BMI of 0 is biologically impossible.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET = "Outcome"

FEATURE_DESCRIPTIONS = {
    "Pregnancies":              "Number of times pregnant",
    "Glucose":                  "Plasma glucose concentration (2h OGTT, mg/dL)",
    "BloodPressure":            "Diastolic blood pressure (mm Hg)",
    "SkinThickness":            "Triceps skin fold thickness (mm)",
    "Insulin":                  "2-Hour serum insulin (μU/mL)",
    "BMI":                      "Body mass index (kg/m²)",
    "DiabetesPedigreeFunction": "Family history likelihood score",
    "Age":                      "Age in years",
}


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes CSV and replace physiological
    zero-values with NaN so the imputer can handle them correctly.

    Args:
        filepath: Path to the diabetes.csv file.

    Returns:
        Cleaned pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download direct CSV: https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv\n"
            "or from Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n"
            "and place it at backend/data/diabetes.csv"
        )

    df = pd.read_csv(filepath)
    print(f"[Preprocess] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Replace impossible zero values with NaN
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    missing_counts = df.isnull().sum()
    print(f"[Preprocess] Missing values after zero replacement:\n{missing_counts[missing_counts > 0]}")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified train/test split that preserves the class ratio in both sets.

    Args:
        df: Cleaned DataFrame from load_and_clean().
        test_size: Fraction of data to reserve for testing.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"[Preprocess] Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"[Preprocess] Train class balance:\n{y_train.value_counts(normalize=True).round(3)}")
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> Pipeline:
    """
    Build a scikit-learn Pipeline that:
      1. Imputes missing values using the column median (robust to outliers)
      2. Scales all features to zero mean and unit variance

    This pipeline object is serialized after fitting and reused at inference time.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def apply_smote(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple:
    """
    Apply SMOTE to balance the training set.
    The Pima dataset has ~65% negative / ~35% positive — moderate imbalance.

    SMOTE creates synthetic minority-class samples by interpolating between
    real minority samples rather than simply duplicating them.

    Args:
        X_train: Scaled training features (numpy array).
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        (X_resampled, y_resampled) — balanced training arrays.
    """
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[Preprocess] After SMOTE — class distribution: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


def prepare_data(filepath: str) -> tuple:
    """
    Full preprocessing pipeline:
        load → clean → split → impute + scale (train only) → SMOTE (train only)

    Args:
        filepath: Path to diabetes.csv.

    Returns:
        (X_train_res, X_test_scaled, y_train_res, y_test, preprocessor)
    """
    df = load_and_clean(filepath)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()

    # ⚠️  .fit_transform() on TRAINING data only
    X_train_scaled = preprocessor.fit_transform(X_train)
    # ⚠️  .transform() ONLY on test data (no fitting)
    X_test_scaled  = preprocessor.transform(X_test)

    # Apply SMOTE after scaling, on training data only
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res, y_test, preprocessor

