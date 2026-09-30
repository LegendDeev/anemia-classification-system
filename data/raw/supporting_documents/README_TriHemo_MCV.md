
# TriHemo-MCV Dataset

## Title
TriHemo-MCV: A Hemoglobin–MCV-Based Tri-Class Dataset for Anemia and Compensated Microcytosis

## Overview
TriHemo-MCV is a curated, de-identified clinical dataset designed to support reproducible research in machine learning-based hematological diagnosis. The dataset enables tri-class classification of anemia-related phenotypes using Complete Blood Count (CBC) parameters and structured lifestyle questionnaires.

The dataset contains 5,037 retrospectively collected patient records from four tertiary-level hospitals in Bangladesh.

## Dataset Characteristics
- Format: CSV
- File Name: tri_hemo_mcv.csv
- Total Records: 5,037
- Total Features: 23
- Target Variable: Class Label (3 classes)

## Clinical Classes and Definitions

| Class ID | Class Name              | Definition                                   | Count |
|----------|-------------------------|-----------------------------------------------|-------|
| 0        | Non-Anemic              | Normal HGB, Normal MCV/MCH                    | 1664  |
| 1        | Anemic                  | Low HGB, Low MCV, High RDW                    | 1732  |
| 2        | Compensated Microcytosis| Normal HGB, Low MCV, High RBC                 | 1641  |

Total Samples: 5,037

## Class Balance
The dataset is well-balanced across the three diagnostic categories, minimizing classification bias and supporting fair model evaluation.

- Non-Anemic: 33.05%
- Anemic: 34.40%
- Compensated Microcytosis: 32.55%

## Data Collection and Ethics
Data were retrospectively collected from four healthcare institutions in Bangladesh between January 22, 2025 and August 25, 2025. Ethical approval was obtained from the Institutional Review Boards (IRB) of participating institutions.

All records were anonymized prior to release. Direct identifiers were removed in compliance with international data protection standards.

Ethics Approval: REC-FSIT-2025/No: 12663

## Annotation and Validation Process
1. Initial labeling by four independent laboratory clinicians.
2. Review of ambiguous cases by senior medical professionals.
3. Final validation based on hemoglobin and MCV thresholds.

This hierarchical annotation protocol ensures high diagnostic reliability.

## File Structure
The dataset is provided as a single file:

- tri_hemo_mcv.csv

No additional folders or metadata files are required.

## Feature Description

### Hematological Parameters

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| Age | Patient age (years) | -- |
| ESR | Erythrocyte Sedimentation Rate | M: 0–15, F: 0–20 mm/hr |
| RBC | Red Blood Cell Count | M: 4.5–5.9, F: 4.1–5.1 ×10¹²/L |
| Platelets | Platelet Count | 150–450 ×10³/µL |
| WBC | White Blood Cell Count | 4.0–10.0 ×10⁹/L |
| Neutrophils | Neutrophil Percentage | 40–75% |
| Lymphocytes | Lymphocyte Percentage | 20–45% |
| Monocytes | Monocyte Percentage | 2–10% |
| Eosinophils | Eosinophil Percentage | 1–6% |
| Basophils | Basophil Percentage | 0–1% |
| HCT | Hematocrit | M: 40–54%, F: 36–48% |
| MCV | Mean Corpuscular Volume | 80–100 fL |
| MCH | Mean Corpuscular Hemoglobin | 27–33 pg |
| MCHC | Mean Corpuscular Hemoglobin Concentration | 32–36 g/dL |
| HGB | Hemoglobin | M: 13–17, F: 12–15 g/dL |
| Gender | Biological Sex | Male/Female |
| MPV | Mean Platelet Volume | 7.5–11.5 fL |
| PCT | Plateletcrit | 0.19–0.39% |
| RDW | Red Cell Distribution Width | 11.5–14.5% |

### Lifestyle Questionnaire Features

| Feature | Description | Values |
|---------|-------------|--------|
| Personal_Anemia_History | Previous anemia history | Yes/No |
| Family_Anemia_History | Family history of anemia | Yes/No |
| Substance_Use | Substance dependency | Yes/No |

## Usage Notes

### Machine Learning
- Target variable: Class
- Input features: All remaining columns
- Recommended splits: 70/15/15 (Train/Validation/Test)

### Federated Learning
Researchers may partition the dataset to simulate institutional data silos.

### Explainable AI
Questionnaire features support SHAP/LIME analysis for interpretability.

## Limitations
- Retrospective data collection
- Geographic limitation to Bangladesh
- Lack of biochemical markers (e.g., Ferritin, B12)

## Ethics and Data Usage
This dataset is approved for retrospective research use. Users must cite this dataset in any derivative work.

Consent was waived due to anonymization.

All procedures comply with the Declaration of Helsinki.


## References
[1] The Application of Machine-Learning Algorithms for Multiclass Classification of Microcytic Anemia, PubMed, 2025.
[2] G. Airlangga, Leveraging Machine Learning for Accurate Anemia Diagnosis, 2025.
[3] Machine Learning-Based Prediction of Hemoglobinopathies, PubMed, 2024.
[4] Dataset Description — Anemia Types Classification, IJIIS, 2025.
[5] Exploring CBC Data for Anemia Diagnosis, BioMedInformatics, 2025.

## Contact
For questions and collaborations, please contact the dataset authors through Md. Hasan Imam Bijoy (hasan15-11743@diu.edu.bd).
