"""
tests/test_preprocessing.py
Unit tests for the data preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import load_and_clean, build_preprocessor, split_data, ZERO_AS_MISSING, FEATURES

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")


@pytest.fixture
def raw_df():
    """Load raw dataset if available, else skip."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("diabetes.csv not found — download it and re-run tests.")
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def cleaned_df():
    if not os.path.exists(DATA_PATH):
        pytest.skip("diabetes.csv not found.")
    return load_and_clean(DATA_PATH)


def test_zero_replacement(cleaned_df):
    """Zero-encoded missing values should become NaN after cleaning."""
    # At least some columns should have NaN after replacing zeros
    assert cleaned_df[ZERO_AS_MISSING].isnull().any().any(), \
        "No NaN values found — zero replacement may not have worked."


def test_no_negative_values(cleaned_df):
    """No feature should have negative values after cleaning."""
    for col in FEATURES:
        non_null = cleaned_df[col].dropna()
        assert (non_null >= 0).all(), f"Negative values found in column '{col}'"


def test_target_column_binary(raw_df):
    """Outcome column must only contain 0 and 1."""
    assert set(raw_df["Outcome"].unique()).issubset({0, 1}), \
        "Target column contains values other than 0 and 1."


def test_split_preserves_ratio(cleaned_df):
    """Stratified split should preserve class ratio within ±5%."""
    X_train, X_test, y_train, y_test = split_data(cleaned_df)
    train_ratio = y_train.mean()
    test_ratio  = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05, \
        f"Class ratio mismatch: train={train_ratio:.3f}, test={test_ratio:.3f}"


def test_preprocessor_no_nan(cleaned_df):
    """After preprocessing, there should be no NaN values."""
    X_train, X_test, y_train, y_test = split_data(cleaned_df)
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled  = preprocessor.transform(X_test)

    assert not np.isnan(X_train_scaled).any(), "NaN found in X_train after preprocessing."
    assert not np.isnan(X_test_scaled).any(),  "NaN found in X_test after preprocessing."


def test_no_data_leakage(cleaned_df):
    """
    Verify that fit() is called only on training data.
    The test set must NOT influence the scaler's mean and variance.
    """
    X_train, X_test, y_train, y_test = split_data(cleaned_df)
    preprocessor = build_preprocessor()

    # Fit on training data
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Scaler's mean should match training set mean (after imputation), not full dataset mean
    scaler    = preprocessor.named_steps["scaler"]
    imputer   = preprocessor.named_steps["imputer"]
    X_imputed = imputer.transform(X_train)

    np.testing.assert_array_almost_equal(
        scaler.mean_,
        X_imputed.mean(axis=0),
        decimal=5,
        err_msg="Scaler mean does not match training set mean — possible data leakage."
    )

