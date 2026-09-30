# Walkthrough: Anemia Classification Using Blood Parameters (CBC)

We have successfully designed, built, trained, and verified the machine learning college project **"Anemia Classification Using Blood Parameters"** in accordance with your college specifications.

---

## 1. Project Deliverables Overview

```
collage_project/
├── data/
│   ├── raw/                  # TriHemo-MCV clinical dataset (5,037 samples) & IRB certificates
│   └── processed/            # Cleaned train.csv, val.csv, test.csv
├── models/                   # Serialized .joblib models and scaler
│   ├── scaler.joblib         # Fitted StandardScaler (leak-free)
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib  # Top performer
│   ├── svm.joblib
│   ├── knn.joblib
│   └── best_model.joblib
├── results/                  # Generated benchmark charts & metrics
│   ├── algorithm_comparison.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── model_benchmark_report.csv
├── src/
│   ├── dataset.py            # Automated download & verification
│   ├── preprocess.py         # Leak-free scaling & stratified splitting
│   ├── validator.py          # 3-Tier input validation & Wintrobe ratio guard
│   ├── models.py             # Architectures for "All Five" ML algorithms
│   ├── train.py              # 5-Fold Stratified Cross-Validation
│   ├── evaluate.py           # Independent test set evaluation & plotting
│   └── predict.py            # CLI and programmatic inference engine
├── templates/
│   └── index.html            # Clinical web dashboard with demo presets
├── static/
│   └── style.css             # Modern clinical UI styling
├── app.py                    # Flask interactive web server
├── run_pipeline.py           # One-click master pipeline runner
├── requirements.txt          # Python dependencies
├── README.md                 # Setup & quick-start guide
├── walkthrough.md            # This walkthrough summary file
├── PROJECT_REPORT.md         # Exhaustive academic documentation for submission
└── VIVA_DEFENSE_GUIDE.md     # 25+ Professor Viva Voce Q&A Cheat Sheet
```

---

## 2. Experimental Benchmark Results (Held-Out Test Set: 756 Samples)

All five canonical algorithms were trained using 5-Fold Stratified Cross-Validation on 3,525 samples and evaluated on the independent test set:

| Algorithm | Accuracy (%) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Multiclass ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **77.91%** | **79.67%** | **77.91%** | **77.53%** | **92.32%** |
| Decision Tree Classifier | 77.78% | 80.33% | 77.78% | 76.94% | 91.26% |
| Support Vector Machine (SVM) | 76.85% | 78.02% | 76.85% | 76.37% | 91.23% |
| Logistic Regression | 76.72% | 76.64% | 76.72% | 76.48% | 91.20% |
| K-Nearest Neighbors (KNN) | 73.54% | 74.12% | 73.54% | 73.59% | 88.36% |

### Clinical Feature Importance (Gini Ranking):
1. **Hemoglobin (HGB): 43.48%** — The dominant threshold marker for general anemia.
2. **Mean Corpuscular Hemoglobin (MCH): 22.11%** — Indicator of cell hypochromia.
3. **Red Blood Cell Count (RBC): 19.04%** — The decisive feature distinguishing Compensated Microcytosis (Thalassemia) from Iron Deficiency Anemia.
4. **Mean Corpuscular Volume (MCV): 12.70%** — Determines microcytic vs normocytic sizing.
5. **MCHC Concentration: 2.67%** — Secondary cellular density check.

---

## 3. Verification & Defensive Safety Tests

We ran automated verification tests for normal cases, clinical edge cases, and invalid inputs:

| Test Case | Inputs | Expected Outcome | Result |
| :--- | :--- | :--- | :---: |
| **Healthy Normal Patient** | `Hb=14.5, RBC=4.8, MCV=88, MCH=30, MCHC=34` | Predicted `Non-Anemic` | **PASS** |
| **Iron-Deficiency Anemia** | `Hb=8.5, RBC=3.2, MCV=66, MCH=19.5, MCHC=28` | Predicted `Anemic` (100% confidence) | **PASS** |
| **Compensated Microcytosis** | `Hb=13.5, RBC=5.8, MCV=70, MCH=23, MCHC=32.8` | Predicted `Compensated Microcytosis` (99.5% confidence) | **PASS** |
| **Negative Input Test** | `Hb=-5.0, RBC=4.8, MCV=88, MCH=30, MCHC=34` | Intercepted with physiological alert: *"Hb cannot be negative"* | **PASS** |
| **Non-numeric String Test** | `Hb='abc', RBC=4.8, MCV=88, MCH=30, MCHC=34` | Intercepted with type alert: *"All measurements must be numeric"* | **PASS** |
| **Wintrobe Ratio Mismatch** | `Hb=14.0, RBC=4.8, MCV=50, MCH=45, MCHC=33` | Flagged biological advisory: *"Reported MCHC deviates from MCH/MCV*100"* | **PASS** |

---

## 4. How to Use & Demonstrate the Project

### A. One-Click Master Pipeline
To re-run the entire pipeline from scratch (download dataset, preprocess, train all 5 models, evaluate, and test):
```powershell
python run_pipeline.py
```

### B. Launch the Interactive Web Dashboard (For Presentations)
```powershell
python app.py
```
Open your browser at **`http://127.0.0.1:5000`**. 
* **Model Selection:** Random Forest is the default engine, but you can select any of the 5 models (**Random Forest, Decision Tree, SVM, Logistic Regression, or KNN**) directly from the dropdown in the form or on the result card header (`Engine: [Dropdown]`).
* **Examiner Presets:** Click the preset buttons (**Normal**, **Anemic**, **Compensated Microcytosis**, or **Invalid Input**) for instant live demonstration in front of your professor!

### C. Run Command-Line Predictions
```powershell
# With default model (Random Forest):
python src/predict.py --hb 14.2 --rbc 4.8 --mcv 87.5 --mch 29.5 --mchc 33.8

# With a specific model choice (e.g. SVM or KNN):
python src/predict.py --hb 14.2 --rbc 4.8 --mcv 87.5 --mch 29.5 --mchc 33.8 --model svm
```

---

## 5. College Submission & Viva Preparation Materials
* **Technical Project Report**: [`PROJECT_REPORT.md`](PROJECT_REPORT.md)
* **Professor Viva Voce Defense Guide**: [`VIVA_DEFENSE_GUIDE.md`](VIVA_DEFENSE_GUIDE.md)

---

## 6. Free Cloud Deployment (Render)
A step-by-step deployment guide has been prepared:
* **Guide**: [`RENDER_DEPLOYMENT_GUIDE.md`](RENDER_DEPLOYMENT_GUIDE.md) explaining how to launch your live public link on Render (`render.com`) completely free without a credit card.
* **Keep-Alive Tip**: Includes instructions to use `cron-job.org` so your app stays 100% awake 24/7 with zero cold-start delay for your professor!


