"""
Master Execution Pipeline: Anemia Classification Using Blood Parameters
========================================================================
Executes the full end-to-end workflow:
  Step 1: Dataset Acquisition & Verification (TriHemo-MCV clinical dataset)
  Step 2: Leak-Free Preprocessing & Stratified Splitting
  Step 3: Model Training & 5-Fold Stratified Cross-Validation (All 5 Algorithms)
  Step 4: Test Set Evaluation & Publication-Quality Visualization Suite
  Step 5: Automated Verification & Clinical Test Cases
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import download_and_extract_dataset
from src.preprocess import preprocess_data
from src.train import train_and_benchmark_models
from src.evaluate import evaluate_all_models
from src.predict import predict_anemia


def run_full_pipeline():
    print("\n" + "=" * 75)
    print("      ANEMIA CLASSIFICATION USING BLOOD PARAMETERS (CBC)")
    print("      END-TO-END MACHINE LEARNING & EVALUATION PIPELINE")
    print("=" * 75 + "\n")

    # Step 1: Ingestion
    print("[1/5] Step 1: Ingesting & Verifying Dataset...")
    csv_path = download_and_extract_dataset()
    print(f"      Verified dataset ready at: {csv_path}\n")

    # Step 2: Preprocessing
    print("[2/5] Step 2: Preprocessing, Scaling & Stratified Splitting...")
    preprocess_data()
    print("      Preprocessing and leak-free StandardScaler complete.\n")

    # Step 3: Model Training & CV
    print("[3/5] Step 3: Training & Benchmarking All 5 ML Algorithms (5-Fold CV)...")
    train_results = train_and_benchmark_models()
    best_model_name = train_results.get("best_model_key", "random_forest")
    print(f"      Best model selected: {best_model_name}\n")

    # Step 4: Evaluation & Visuals
    print("[4/5] Step 4: Held-Out Test Set Evaluation & Plot Generation...")
    report_df = evaluate_all_models()
    print("      Generated benchmark tables and 4 visual charts in results/.\n")

    # Step 5: Verification & Safety Tests
    print("[5/5] Step 5: Running Verification & Defensive Layer Tests...")
    
    # Test 1: Normal Sample
    r1 = predict_anemia(14.5, 4.8, 88.0, 30.0, 34.0)
    assert r1["success"] and r1["predicted_category"] == "Non-Anemic", f"Normal test failed: {r1}"
    print("      [PASS] Test 1: Healthy Normal Patient -> Predicted 'Non-Anemic'")

    # Test 2: Anemic Sample
    r2 = predict_anemia(8.5, 3.2, 66.0, 19.5, 28.0)
    assert r2["success"] and r2["predicted_category"] == "Anemic", f"Anemic test failed: {r2}"
    print("      [PASS] Test 2: Severe Iron-Deficiency Anemia -> Predicted 'Anemic'")

    # Test 3: Compensated Microcytosis
    r3 = predict_anemia(13.5, 5.8, 70.0, 23.0, 32.8)
    assert r3["success"] and r3["predicted_category"] == "Compensated Microcytosis", f"Microcytosis test failed: {r3}"
    print("      [PASS] Test 3: Compensated Microcytosis -> Predicted 'Compensated Microcytosis'")

    # Test 4: Defensive Interception (Negative Input)
    r4 = predict_anemia(-5.0, 4.8, 88.0, 30.0, 34.0)
    assert not r4["success"] and "cannot be zero or negative" in r4["error"], f"Negative test failed: {r4}"
    print("      [PASS] Test 4: Defensive Validation Intercepted Negative Input (-5.0 g/dL)")

    print("\n" + "=" * 75)
    print("                   PIPELINE EXECUTION COMPLETE!")
    print("=" * 75)
    print("  Artifacts Generated:")
    print("    - Models:   models/*.joblib (All 5 algorithms + scaler + best_model)")
    print("    - Data:     data/processed/{train,val,test}.csv")
    print("    - Visuals:  results/*.png (Confusion matrices, ROC, Feature importance, Comparison)")
    print("    - Report:   results/model_benchmark_report.csv")
    print("\n  To Launch the Interactive Web Dashboard:")
    print("    python app.py  (Opens at http://127.0.0.1:5000)")
    print("\n  To Run CLI Predictions:")
    print("    python src/predict.py --hb 14.0 --rbc 4.8 --mcv 88.0 --mch 30.0 --mchc 34.0")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_full_pipeline()
