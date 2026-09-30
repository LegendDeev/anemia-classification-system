"""
Comprehensive Evaluation & Benchmarking Visualization Suite
============================================================
Evaluates all 5 trained models on the held-out test partition.
Computes Accuracy, Precision, Recall, F1, and Multiclass ROC-AUC.
Generates publication-quality charts and confusion matrices in results/.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.preprocess import (
    MODELS_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, FEATURE_COLUMNS
)
from src.models import MODEL_METADATA

RESULTS_DIR = PROJECT_ROOT / "results"


def load_test_data():
    """Loads preprocessed test partition and fitted scaler."""
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    
    X_test_raw = test_df[FEATURE_COLUMNS].values
    X_test_scaled = scaler.transform(X_test_raw)
    y_test = test_df["label"].values
    return X_test_scaled, y_test, test_df


def evaluate_all_models() -> pd.DataFrame:
    """
    Evaluates all 5 serialized models on the test set, generates plots and CSV reports.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test, test_df = load_test_data()

    model_keys = list(MODEL_METADATA.keys())
    metrics_list = []
    confusion_matrices = {}
    y_probs_dict = {}

    print("=" * 80)
    print("           ANEMIA CLASSIFICATION - INDEPENDENT TEST SET EVALUATION")
    print("=" * 80)
    print(f"{'Algorithm':<28} | {'Accuracy':<9} | {'Precision':<10} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC'}")
    print("-" * 80)

    # Binarize labels for multiclass ROC-AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = len(CLASS_NAMES)

    for key in model_keys:
        model_path = MODELS_DIR / f"{key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Run src/train.py first.")

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            try:
                roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr") * 100
            except Exception:
                roc_auc = 0.0
        else:
            y_prob = None
            roc_auc = 0.0

        y_probs_dict[key] = y_prob
        confusion_matrices[key] = confusion_matrix(y_test, y_pred)

        display_name = MODEL_METADATA[key]["name"]
        print(f"{display_name:<28} | {acc:5.2f}%   | {prec:5.2f}%    | {rec:5.2f}%   | {f1:5.2f}%   | {roc_auc:5.2f}%")

        metrics_list.append({
            "Algorithm": display_name,
            "Model Key": key,
            "Accuracy (%)": round(acc, 2),
            "Precision (%)": round(prec, 2),
            "Recall (%)": round(rec, 2),
            "F1-Score (%)": round(f1, 2),
            "ROC-AUC (%)": round(roc_auc, 2)
        })

    print("-" * 80)
    report_df = pd.DataFrame(metrics_list)
    report_csv_path = RESULTS_DIR / "model_benchmark_report.csv"
    report_df.to_csv(report_csv_path, index=False)
    print(f"[*] Benchmark report saved to: {report_csv_path}")

    # Generate Visualizations
    _plot_algorithm_comparison(report_df)
    _plot_confusion_matrices(confusion_matrices)
    _plot_roc_curves(y_test_bin, y_probs_dict)
    _plot_feature_importance()

    return report_df


def _plot_algorithm_comparison(report_df: pd.DataFrame):
    """Generates grouped bar chart comparing all 5 algorithms."""
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    df_melted = pd.melt(
        report_df,
        id_vars=["Algorithm"],
        value_vars=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"],
        var_name="Metric",
        value_name="Score"
    )

    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    ax = sns.barplot(data=df_melted, x="Algorithm", y="Score", hue="Metric", palette=palette)

    plt.title("Performance Comparison of All 5 Classification Algorithms on Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score (%)", fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.ylim(60, 95)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_path = RESULTS_DIR / "algorithm_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Generated algorithm comparison chart: {out_path}")


def _plot_confusion_matrices(cms: Dict[str, np.ndarray]):
    """Generates a 2x3 grid of confusion matrix heatmaps for all 5 models."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    model_keys = list(MODEL_METADATA.keys())
    cm_palettes = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

    for idx, key in enumerate(model_keys):
        cm = cms[key]
        display_name = MODEL_METADATA[key]["name"]
        ax = axes[idx]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cm_palettes[idx],
            xticklabels=["Non-Anemic", "Anemic", "Compensated"],
            yticklabels=["Non-Anemic", "Anemic", "Compensated"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{display_name}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    # Hide the 6th empty subplot
    axes[5].axis("off")
    fig.suptitle("Confusion Matrices Across All 5 Algorithms (Held-Out Test Set)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out_path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Generated confusion matrices chart: {out_path}")


def _plot_roc_curves(y_test_bin: np.ndarray, y_probs_dict: Dict[str, np.ndarray]):
    """Plots multiclass ROC curves for top-performing model and comparison."""
    # Read best model metadata
    best_meta_path = MODELS_DIR / "best_model_meta.joblib"
    best_key = "random_forest"
    if best_meta_path.exists():
        meta = joblib.load(best_meta_path)
        best_key = meta.get("best_model_key", "random_forest")

    y_prob = y_probs_dict.get(best_key)
    if y_prob is None:
        return

    plt.figure(figsize=(9, 7))
    sns.set_theme(style="whitegrid")
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]

    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr,
            color=colors[i],
            lw=2.5,
            label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(f"Multiclass ROC Curves - Best Model: {MODEL_METADATA[best_key]['name']}", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()

    out_path = RESULTS_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Generated ROC curves chart: {out_path}")


def _plot_feature_importance():
    """Plots feature importance of the 5 blood parameters from Random Forest."""
    rf_path = MODELS_DIR / "random_forest.joblib"
    if not rf_path.exists():
        return

    rf_model = joblib.load(rf_path)
    importances = rf_model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLUMNS[i] for i in indices]
    sorted_importances = [importances[i] * 100 for i in indices]

    feature_full_names = {
        "HGB": "Hemoglobin (HGB)",
        "RBC": "RBC Count (RBC)",
        "MCV": "Mean Cell Volume (MCV)",
        "MCH": "Mean Cell Hemoglobin (MCH)",
        "MCHC": "MCHC Concentration"
    }
    sorted_labels = [feature_full_names.get(f, f) for f in sorted_features]

    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("mako", len(sorted_features))

    ax = sns.barplot(x=sorted_importances, y=sorted_labels, hue=sorted_labels, palette=palette, legend=False)
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{width:.1f}%", (width + 0.5, p.get_y() + p.get_height() / 2),
                    ha="left", va="center", fontsize=11, fontweight="bold")

    plt.title("Clinical Feature Importance Ranking (Random Forest)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Gini Feature Importance (%)", fontsize=12)
    plt.xlim(0, max(sorted_importances) + 8)
    plt.tight_layout()

    out_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Generated feature importance chart: {out_path}")


if __name__ == "__main__":
    evaluate_all_models()
