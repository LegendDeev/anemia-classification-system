# Anemia Classification Using Blood Parameters (CBC)

[live link](https://anemia-classification-system-njx8.onrender.com/predict)

An end-to-end Machine Learning college project for automated clinical classification of anemia-related phenotypes from Complete Blood Count (CBC) laboratory measurements.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-TriHemo--MCV%20(5%2C037%20Samples)-green.svg)](https://doi.org/10.6084/m9.figshare.27986873)
[![License](https://img.shields.io/badge/License-Academic%20Use-orange.svg)]()

---

## 📌 Project Overview

This project classifies patient blood profiles into three clinically verified diagnostic categories:
1. **Non-Anemic**: Normal hemoglobin and standard red cell volume.
2. **Anemic**: Depressed hemoglobin and altered RBC morphology (iron-deficiency, acute loss, or chronic disease).
3. **Compensated Microcytosis**: Normal hemoglobin with elevated RBC count and low cell volume (typical in Thalassemia trait).

### Features Utilized (5 Core Blood Parameters):
* **Hemoglobin (HGB)** (g/dL)
* **Red Blood Cell Count (RBC)** (million/µL or M/µL)
* **Mean Corpuscular Volume (MCV)** (fL)
* **Mean Corpuscular Hemoglobin (MCH)** (pg)
* **Mean Corpuscular Hemoglobin Concentration (MCHC)** (g/dL)

### "All Five" Supervised Machine Learning Algorithms Implemented:
* **Logistic Regression** (Multinomial baseline)
* **Decision Tree Classifier** (CART with Gini Impurity)
* **Random Forest Classifier** (Bagging Ensemble — **Best Performing**)
* **Support Vector Machine (SVM)** (RBF Kernel)
* **K-Nearest Neighbors (KNN)** (Distance-Weighted Metric)

---

## 🛡️ Defensive Input Validation Layer (`src/validator.py`)
Machine learning models often fail silently or crash when fed negative numbers, non-numeric strings, or physically impossible measurements. This project incorporates a **three-tier clinical validation guard**:
* **Tier 1 (Type & Null Interception)**: Prevents non-numeric values (`"abc"`, empty fields) from causing unhandled Python crashes.
* **Tier 2 (Physiological Reference Bounds)**: Rejects negative values, zero, or humanly impossible numbers (e.g. $Hb \notin [2.0, 25.0]$ g/dL).
* **Tier 3 (Biological Wintrobe Consistency)**: Computes expected $MCHC = (MCH / MCV) \times 100$ and flags conflicting laboratory measurements.

---

## 📂 Project Directory Structure

```text
collage_project/
├── data/
│   ├── raw/                  # Downloaded TriHemo-MCV clinical dataset & certificates
│   └── processed/            # Cleaned 5-feature train/val/test CSV splits
├── models/                   # Serialized .joblib models and scaler
│   ├── scaler.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── knn.joblib
│   └── best_model.joblib
├── results/                  # Generated benchmark plots & CSV reports
│   ├── algorithm_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── model_benchmark_report.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py            # Dataset acquisition & verification
│   ├── preprocess.py         # Leak-free StandardScaler & stratified splitting
│   ├── validator.py          # Input validation, bounds checks & sanitization
│   ├── models.py             # Definitions of all 5 ML algorithms
│   ├── train.py              # 5-Fold Stratified Cross-Validation & training
│   ├── evaluate.py           # Independent test set evaluation & plotting
│   └── predict.py            # CLI and programmatic inference engine
├── templates/
│   └── index.html            # Web UI template with presets & analytics
├── static/
│   └── style.css             # UI styling
├── app.py                    # Flask Web Application
├── run_pipeline.py           # One-click master pipeline runner
├── requirements.txt          # Python dependencies
├── PROJECT_REPORT.md         # Full Academic Documentation for submission
└── VIVA_DEFENSE_GUIDE.md     # 25+ Professor Viva Voce Q&A Cheat Sheet
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Executes dataset download, leak-free preprocessing, 5-Fold Stratified Cross-Validation across all 5 models, test evaluation, and chart generation:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Web Dashboard
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser. Try clicking the **Examiner Demo Presets** for instant live classifications!

### 4. Run CLI Predictions
```bash
# Valid patient run:
python src/predict.py --hb 14.2 --rbc 4.8 --mcv 87.5 --mch 29.5 --mchc 33.8

# Testing invalid input interception (defensive safety test):
python src/predict.py --hb -5.0 --rbc 4.8 --mcv 87.5 --mch 29.5 --mchc 33.8
```

---

## 📊 Experimental Results (Held-Out Test Set)

| Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Multiclass ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **77.91%** | **79.67%** | **77.91%** | **77.53%** | **92.32%** |
| Decision Tree Classifier | 77.78% | 80.33% | 77.78% | 76.94% | 91.26% |
| Support Vector Machine (SVM) | 76.85% | 78.02% | 76.85% | 76.37% | 91.23% |
| Logistic Regression | 76.72% | 76.64% | 76.72% | 76.48% | 91.20% |
| K-Nearest Neighbors (KNN) | 73.54% | 74.12% | 73.54% | 73.59% | 88.36% |

