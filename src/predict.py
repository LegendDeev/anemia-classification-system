"""
Inference & Prediction Engine with Defensive Validation
========================================================
Accepts raw CBC blood measurements, runs them through the defensive
validator, scales inputs, and generates predictions across all 5 models.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from src.validator import validate_blood_parameters, get_parameter_status, PHYSIOLOGICAL_BOUNDS
from src.preprocess import MODELS_DIR, CLASS_NAMES, FEATURE_COLUMNS
from src.models import MODEL_METADATA


def predict_anemia(
    hgb: Union[float, int, str],
    rbc: Union[float, int, str],
    mcv: Union[float, int, str],
    mch: Union[float, int, str],
    mchc: Union[float, int, str],
    model_choice: str = "random_forest"
) -> Dict[str, Any]:
    """
    Main prediction pipeline:
      1. Validates inputs through 3-tier clinical validator.
      2. Scales 5 parameters using training scaler.
      3. Predicts using the selected model (default: random_forest).
      4. Queries all 5 models for algorithm consensus.
      5. Gathers clinical parameter indicators.
    """
    # Tier 1 & 2: Defensive Validation
    is_valid, clean_or_err, warnings = validate_blood_parameters(hgb, rbc, mcv, mch, mchc)
    if not is_valid:
        return {
            "success": False,
            "error": clean_or_err,
            "warnings": []
        }

    cleaned_data = clean_or_err

    # Load Scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    if not scaler_path.exists():
        return {
            "success": False,
            "error": "Trained scaler not found. Please train models first using run_pipeline.py.",
            "warnings": []
        }
    scaler = joblib.load(scaler_path)

    # Format vector [1, 5]
    feature_vector = np.array([[
        cleaned_data["HGB"],
        cleaned_data["RBC"],
        cleaned_data["MCV"],
        cleaned_data["MCH"],
        cleaned_data["MCHC"]
    ]])
    scaled_vector = scaler.transform(feature_vector)

    # Determine Active Model (Default: Random Forest)
    valid_keys = list(MODEL_METADATA.keys())
    if model_choice not in valid_keys:
        model_choice = "random_forest"

    active_model_path = MODELS_DIR / f"{model_choice}.joblib"
    if not active_model_path.exists():
        active_model_path = MODELS_DIR / "best_model.joblib"
        model_choice = "random_forest"

    active_model_name = MODEL_METADATA.get(model_choice, {}).get("name", "Random Forest Classifier")
    active_model = joblib.load(active_model_path)
    pred_idx = int(active_model.predict(scaled_vector)[0])
    predicted_category = CLASS_NAMES[pred_idx]

    # Probabilities
    if hasattr(active_model, "predict_proba"):
        probs = active_model.predict_proba(scaled_vector)[0]
        confidence = float(probs[pred_idx]) * 100
        prob_breakdown = {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    else:
        confidence = 100.0
        prob_breakdown = {c: (100.0 if c == predicted_category else 0.0) for c in CLASS_NAMES}

    # Query all 5 models for consensus comparison
    all_models_consensus = {}
    for key, meta_info in MODEL_METADATA.items():
        m_path = MODELS_DIR / f"{key}.joblib"
        if m_path.exists():
            m_inst = joblib.load(m_path)
            p_idx = int(m_inst.predict(scaled_vector)[0])
            all_models_consensus[meta_info["name"]] = CLASS_NAMES[p_idx]

    # Clinical parameter status analysis
    parameter_analysis = {}
    for param in FEATURE_COLUMNS:
        parameter_analysis[param] = {
            "value": cleaned_data[param],
            "unit": PHYSIOLOGICAL_BOUNDS[param]["unit"],
            "name": PHYSIOLOGICAL_BOUNDS[param]["name"],
            **get_parameter_status(param, cleaned_data[param])
        }

    return {
        "success": True,
        "input_parameters": cleaned_data,
        "predicted_category": predicted_category,
        "confidence_percentage": round(confidence, 2),
        "class_probabilities": prob_breakdown,
        "best_model_used": active_model_name,
        "selected_model_key": model_choice,
        "all_models_predictions": all_models_consensus,
        "parameter_analysis": parameter_analysis,
        "warnings": warnings
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict Anemia category using 5 Complete Blood Count (CBC) measurements."
    )
    parser.add_argument("--hb", type=str, required=True, help="Hemoglobin (g/dL), e.g. 13.5")
    parser.add_argument("--rbc", type=str, required=True, help="Red Blood Cell count (M/µL), e.g. 4.8")
    parser.add_argument("--mcv", type=str, required=True, help="Mean Corpuscular Volume (fL), e.g. 85.0")
    parser.add_argument("--mch", type=str, required=True, help="Mean Corpuscular Hemoglobin (pg), e.g. 29.0")
    parser.add_argument("--mchc", type=str, required=True, help="Mean Corpuscular Hemoglobin Concentration (g/dL), e.g. 33.5")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "logistic_regression", "decision_tree", "svm", "knn"],
                        help="Choose classification engine (default: random_forest)")

    args = parser.parse_args()

    result = predict_anemia(args.hb, args.rbc, args.mcv, args.mch, args.mchc, model_choice=args.model)

    if not result["success"]:
        print("\n" + "!" * 65)
        print("  [X] VALIDATION ERROR (Defensive Layer Interception):")
        print(f"      {result['error']}")
        print("!" * 65 + "\n")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("           CLINICAL ANEMIA CLASSIFICATION REPORT")
    print("=" * 65)
    print(f"[*] Primary Diagnosis:        {result['predicted_category'].upper()}")
    print(f"[*] Model Confidence:         {result['confidence_percentage']:.2f}%")
    print(f"[*] Evaluated with:           {result['best_model_used']}")
    print("-" * 65)
    print("  Class Probability Distribution:")
    for cls_name, prob in result["class_probabilities"].items():
        bar = "#" * int(prob / 5)
        print(f"    - {cls_name:<25}: {prob:5.2f}% | {bar}")

    print("-" * 65)
    print("  All 5 Machine Learning Models Consensus:")
    for m_name, pred_cat in result["all_models_predictions"].items():
        match_symbol = "[MATCH]" if pred_cat == result["predicted_category"] else "[DIFF] "
        print(f"    - {m_name:<30}: {pred_cat} {match_symbol}")

    print("-" * 65)
    print("  Blood Parameters Clinical Status:")
    for param, info in result["parameter_analysis"].items():
        print(f"    - {info['name']} ({param}): {info['value']} {info['unit']} -> [{info['status']}] {info['message']}")

    if result["warnings"]:
        print("-" * 65)
        print("  Clinical Advisory Warnings:")
        for w in result["warnings"]:
            print(f"    [!] {w}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
