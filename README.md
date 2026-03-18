# 🏥 Patient Readmission Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Best AUC](https://img.shields.io/badge/Best_AUC-0.82-brightgreen?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)]()
[![Dataset](https://img.shields.io/badge/Dataset-Synthetic_EHR-orange?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> **Predicts whether a hospital patient will be readmitted within 30 days of discharge.**
> Built to support NHS discharge planning decisions using machine learning on Electronic Health Record (EHR) data.

📊 [View Results](#-results) · 🔬 [Methodology](#-how-it-works) · 🚀 [Run Locally](#-installation)

---

## ⚠️ Data Note

This project uses **synthetic EHR data** (3,000 patients) generated to statistically mirror the properties of real clinical datasets.

> Real clinical databases such as MIMIC-III require credentialed access via PhysioNet.
> Apply here: [physionet.org/settings/credentialing](https://physionet.org/settings/credentialing/)

---

## 🎯 The Problem This Solves

Unplanned hospital readmissions cost the NHS approximately **£1.6 billion per year**, and research shows up to 30% could be prevented with better discharge planning.

When a patient is discharged, clinical teams must decide:
- Does this patient need early outpatient follow-up?
- Should a community nurse visit be arranged?
- Does this patient need enhanced medication support?

Currently, these decisions rely on clinical intuition alone. **This model provides an objective, data-driven risk score** so limited follow-up resources go to the patients who need them most.

---

## 🔬 How It Works

### Step 1 — 15 Clinical Features from EHR Data

| Feature | What It Measures |
|---|---|
| Age | Patient age at admission |
| Number of diagnoses | Total active conditions |
| Number of medications | Drug complexity |
| Length of stay | Days in hospital |
| Previous admissions | Readmission history (strongest predictor) |
| Emergency admission | Planned vs emergency (0/1) |
| Charlson score | Comorbidity burden |
| Creatinine | Kidney function marker |
| Hemoglobin | Anaemia indicator |
| Glucose | Blood sugar level |
| Sodium / Potassium | Electrolyte balance |

### Step 2 — Three Models Compared

| Model | AUC-ROC | F1-Score |
|---|---|---|
| Logistic Regression | 0.78 | 0.71 |
| Random Forest | 0.80 | 0.73 |
| **✅ Gradient Boosting** | **0.82** | **0.75** |

**Gradient Boosting** performed best — it correctly ranks a readmitted patient above a non-readmitted patient **82% of the time**.

### Step 3 — Top Risk Factors Identified

1. **Number of previous admissions** — strongest single predictor
2. **Creatinine level** — elevated = kidney dysfunction risk
3. **Charlson comorbidity score** — multiple chronic conditions
4. **Age** — older patients at higher risk
5. **Hemoglobin** — low levels (anaemia) increase risk

These findings match published clinical literature on readmission risk factors.

---

## 📊 Results

| Metric | Value |
|---|---|
| Total patients | 3,000 |
| Readmission rate | ~35% |
| Best model | Gradient Boosting |
| Best AUC-ROC | **0.82** |
| Best F1-Score | **0.75** |
| Features used | 15 |
| Top predictor | Number of previous admissions |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/omiiii274/patient-readmission-predictor
cd patient-readmission-predictor

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
```

---

## 🛠 Tools Used

| Category | Tools |
|---|---|
| ML Models | Logistic Regression, Random Forest, Gradient Boosting |
| ML Framework | scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib |
| Evaluation | ROC-AUC, F1-Score, Feature Importance |

---

## 📁 Project Structure

```
patient-readmission-predictor/
│
├── main.py                    # Main model training and evaluation script
├── requirements.txt           # Python dependencies
├── feature_importance.png     # Feature importance chart
├── model_comparison.png       # ROC curve comparison across models
├── readmission_by_age.png     # Readmission rate by age group
├── roc_curves.png             # Full ROC curve visualisation
└── README.md
```

---

## 👤 Author

**Omkar Salekar** — MSc Data Science & AI, Oxford Brookes University
PSI Member · IEEE Published Researcher

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/omkar-salekar-23188026b)
