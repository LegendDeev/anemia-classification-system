"""
Data Preprocessing & Splitting Pipeline
=======================================
Selects the required 5 CBC parameters (HGB, RBC, MCV, MCH, MCHC),
performs leak-free stratified splitting (70% train, 15% val, 15% test),
fits StandardScaler strictly on training partition, and saves preprocessed artifacts.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.dataset import load_raw_dataset
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# The 5 college-specified clinical features
FEATURE_COLUMNS = ["HGB", "RBC", "MCV", "MCH", "MCHC"]
TARGET_COLUMN = "Class"

CLASS_NAMES = ["Non-Anemic", "Anemic", "Compensated Microcytosis"]
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INT_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def preprocess_data(
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Executes leak-free preprocessing:
      1. Filters to the 5 designated blood parameters.
      2. Encodes classes: 0 -> Non-Anemic, 1 -> Anemic, 2 -> Compensated Microcytosis.
      3. Stratified split into Train (70%), Val (15%), Test (15%).
      4. Fits StandardScaler on X_train only and transforms all partitions.
      5. Saves the fitted scaler and CSV datasets.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_dataset()

    # Extract 5 features + target
    df = df_raw[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Verify no missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"[!] Imputing {missing_count} missing values with median...")
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    # Map labels to integers
    df["label"] = df[TARGET_COLUMN].map(CLASS_TO_INT)
    if df["label"].isnull().any():
        unknown_classes = df[df["label"].isnull()][TARGET_COLUMN].unique()
        raise ValueError(f"Encountered unexpected class labels: {unknown_classes}")

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    # Step 1: Split into Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Step 2: Split Train+Val into Train (70% total) and Val (15% total)
    # val_size_adjusted = 0.15 / 0.85 approx 0.1765
    val_size_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        stratify=y_train_val,
        random_state=random_state
    )

    print(f"[*] Stratified Dataset Split completed:")
    print(f"    - Training samples:   {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    - Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    - Test samples:       {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Fit Scaler strictly on X_train to prevent Data Leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Persist the fitted scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[*] Saved fitted StandardScaler to: {scaler_path}")

    # Export processed CSVs for transparency and inspection
    train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    train_df["label"] = y_train
    train_df["Class"] = train_df["label"].map(INT_TO_CLASS)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)

    val_df = pd.DataFrame(X_val, columns=FEATURE_COLUMNS)
    val_df["label"] = y_val
    val_df["Class"] = val_df["label"].map(INT_TO_CLASS)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)

    test_df = pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
    test_df["label"] = y_test
    test_df["Class"] = test_df["label"].map(INT_TO_CLASS)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    print(f"[*] Saved processed train, val, and test CSV partitions to: {PROCESSED_DATA_DIR}")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    preprocess_data()
