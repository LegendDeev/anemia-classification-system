"""
Model Training & 5-Fold Stratified Cross-Validation Engine
==========================================================
Trains all 5 canonical algorithms, benchmarks them via 5-Fold Stratified CV,
evaluates on validation split, and serializes trained models to disk.
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import preprocess_data, MODELS_DIR, CLASS_NAMES
from src.models import get_all_models, MODEL_METADATA


def train_and_benchmark_models() -> Dict[str, Any]:
    """
    Executes 5-Fold Cross Validation and training for all 5 algorithms.
    Identifies the best model and serializes all models to models/ directory.
    """
    print("=" * 70)
    print("      ANEMIA CLASSIFICATION - MODEL TRAINING & 5-FOLD CV PIPELINE")
    print("=" * 70)

    # Load preprocessed partitions
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data()
    models = get_all_models(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    training_results = []
    trained_model_instances = {}

    print("\n[*] Starting 5-Fold Stratified Cross-Validation & Model Training...")
    print("-" * 70)
    print(f"{'Algorithm':<28} | {'5-Fold CV Acc (Mean ± Std)':<22} | {'Val Acc':<9} | {'Val F1'}")
    print("-" * 70)

    best_val_score = -1.0
    best_model_name = ""

    for model_key, model in models.items():
        meta = MODEL_METADATA[model_key]
        display_name = meta["name"]

        # 1. 5-Fold Stratified Cross-Validation on X_train
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_mean = cv_scores.mean() * 100
        cv_std = cv_scores.std() * 100

        # 2. Fit on full training set
        model.fit(X_train, y_train)
        trained_model_instances[model_key] = model

        # 3. Evaluate on validation set
        val_preds = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds) * 100
        val_f1 = f1_score(y_val, val_preds, average="weighted") * 100

        print(f"{display_name:<28} | {cv_mean:5.2f}% ± {cv_std:4.2f}%          | {val_acc:5.2f}%   | {val_f1:5.2f}%")

        # 4. Serialize individual model
        model_save_path = MODELS_DIR / f"{model_key}.joblib"
        joblib.dump(model, model_save_path)

        training_results.append({
            "model_key": model_key,
            "display_name": display_name,
            "cv_mean_accuracy": cv_mean,
            "cv_std_accuracy": cv_std,
            "val_accuracy": val_acc,
            "val_f1_weighted": val_f1
        })

        if val_f1 > best_val_score:
            best_val_score = val_f1
            best_model_name = model_key

    print("-" * 70)
    print(f"[*] Top Performing Model on Validation Set: '{MODEL_METADATA[best_model_name]['name']}' ({best_val_score:.2f}% F1)")
    
    # Save the designated best model
    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(trained_model_instances[best_model_name], best_model_path)
    
    # Save metadata indicating which algorithm won
    best_meta_path = MODELS_DIR / "best_model_meta.joblib"
    joblib.dump({
        "best_model_key": best_model_name,
        "best_model_name": MODEL_METADATA[best_model_name]["name"],
        "class_names": CLASS_NAMES,
        "val_score": best_val_score
    }, best_meta_path)

    print(f"[*] Saved best model copy to: {best_model_path}")
    print(f"[*] All 5 trained models saved to: {MODELS_DIR}\n")

    return {
        "results": training_results,
        "best_model_key": best_model_name,
        "models": trained_model_instances
    }


if __name__ == "__main__":
    train_and_benchmark_models()
