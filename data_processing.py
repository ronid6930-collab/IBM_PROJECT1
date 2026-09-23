"""
data_processing.py
------------------
Handles loading, cleaning, preprocessing, and encoding of the
job_salary_prediction_dataset.csv dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "job_salary_prediction_dataset.csv")
MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Column constants ──────────────────────────────────────────────────────────
TARGET_COL = "salary"

CATEGORICAL_COLS = [
    "job_title", "education_level", "industry",
    "company_size", "location", "remote_work",
]

NUMERICAL_COLS = ["experience_years", "skills_count", "certifications"]

FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS   # model input columns


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV dataset and return a DataFrame."""
    df = pd.read_csv(path)
    return df


# ── 2. Clean ──────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Drop exact duplicate rows.
    • Drop rows where the target salary is missing.
    • Fill remaining numeric NaNs with median; categorical NaNs with mode.
    """
    df = df.drop_duplicates()

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Fill missing numerical values with median
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Ensure correct dtypes
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    return df.reset_index(drop=True)


# ── 3. Encode ─────────────────────────────────────────────────────────────────
def encode_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode every categorical column.

    Parameters
    ----------
    df       : cleaned DataFrame
    encoders : pre-fitted encoders dict  (pass when predicting, None when training)
    fit      : True  → fit new encoders and return them
               False → use the supplied encoders to transform

    Returns
    -------
    df_enc   : DataFrame with encoded categorical columns
    encoders : dict[col_name -> LabelEncoder]
    """
    df_enc = df.copy()

    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            if col in df_enc.columns:
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                encoders[col] = le
    else:
        if encoders is None:
            raise ValueError("encoders must be provided when fit=False")
        for col in CATEGORICAL_COLS:
            if col in df_enc.columns:
                le = encoders[col]
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df_enc[col] = df_enc[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df_enc[col] = le.transform(df_enc[col])

    return df_enc, encoders


# ── 4. Prepare X / y ─────────────────────────────────────────────────────────
def get_features_target(
    df_enc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split encoded DataFrame into feature matrix X and target y."""
    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]
    return X, y


# ── 5. Save / load encoders ───────────────────────────────────────────────────
def save_encoders(encoders: dict, path: str | None = None) -> str:
    if path is None:
        path = os.path.join(MODELS_DIR, "encoders.pkl")
    joblib.dump(encoders, path)
    return path


def load_encoders(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(MODELS_DIR, "encoders.pkl")
    return joblib.load(path)


# ── 6. Full pipeline (used by model_training.py) ──────────────────────────────
def run_pipeline(path: str = DATASET_PATH):
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    X        : feature DataFrame (encoded, ready for sklearn)
    y        : salary Series
    df_raw   : cleaned but NOT encoded DataFrame (useful for EDA)
    encoders : fitted LabelEncoders
    """
    df_raw   = load_data(path)
    df_clean = clean_data(df_raw)
    df_enc, encoders = encode_features(df_clean, fit=True)
    X, y = get_features_target(df_enc)
    return X, y, df_clean, encoders


# ── 7. Single-row encoder (used by Streamlit app) ────────────────────────────
def encode_single_row(row_dict: dict, encoders: dict) -> pd.DataFrame:
    """
    Encode a single input dict from the Streamlit form.

    Parameters
    ----------
    row_dict : {feature_name: value, ...}
    encoders : pre-fitted encoders

    Returns
    -------
    pd.DataFrame with one row, columns = FEATURE_COLS
    """
    df_row = pd.DataFrame([row_dict])
    df_enc, _ = encode_features(df_row, encoders=encoders, fit=False)
    return df_enc[FEATURE_COLS]
