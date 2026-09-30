"""
Clinical Anemia Classification Web Application (Flask)
======================================================
Serves an interactive medical dashboard for Complete Blood Count (CBC) analysis,
real-time multi-model consensus prediction, and evaluation metrics visualizer.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, render_template, request, jsonify, send_from_directory
from src.predict import predict_anemia

app = Flask(__name__)
RESULTS_DIR = PROJECT_ROOT / "results"


@app.route("/")
def index():
    """Renders the main clinical dashboard."""
    return render_template(
        "index.html",
        result=None,
        inputs=None,
        error=None,
        warnings=[],
        selected_model="random_forest"
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handles web form submission for patient blood evaluation."""
    raw_inputs = {
        "HGB": request.form.get("hgb", ""),
        "RBC": request.form.get("rbc", ""),
        "MCV": request.form.get("mcv", ""),
        "MCH": request.form.get("mch", ""),
        "MCHC": request.form.get("mchc", "")
    }
    model_choice = request.form.get("model_choice", "random_forest")

    result = predict_anemia(
        hgb=raw_inputs["HGB"],
        rbc=raw_inputs["RBC"],
        mcv=raw_inputs["MCV"],
        mch=raw_inputs["MCH"],
        mchc=raw_inputs["MCHC"],
        model_choice=model_choice
    )

    if not result["success"]:
        return render_template(
            "index.html",
            result=None,
            inputs=raw_inputs,
            error=result["error"],
            warnings=[],
            selected_model=model_choice
        )

    return render_template(
        "index.html",
        result=result,
        inputs=raw_inputs,
        error=None,
        warnings=result.get("warnings", []),
        selected_model=model_choice
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be a valid JSON object."}), 400

    hgb = data.get("hgb") or data.get("HGB")
    rbc = data.get("rbc") or data.get("RBC")
    mcv = data.get("mcv") or data.get("MCV")
    mch = data.get("mch") or data.get("MCH")
    mchc = data.get("mchc") or data.get("MCHC")
    model_choice = data.get("model") or data.get("model_choice", "random_forest")

    result = predict_anemia(hgb=hgb, rbc=rbc, mcv=mcv, mch=mch, mchc=mchc, model_choice=model_choice)
    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serves generated evaluation plots and charts."""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  [+] Starting Anemia Classification Clinical Web Server...")
    print("  [+] Local URL: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
