

# 🏥 Patient Readmission Risk Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![AUC](https://img.shields.io/badge/Best_AUC-0.82-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## 📋 About This Project

This project predicts whether a hospital patient will be readmitted within 30 days of discharge using machine learning. It analyses 15 clinical features from Electronic Health Records (EHR) — including age, previous admissions, lab values, and comorbidity scores — to generate a risk score for each patient.

Unplanned hospital readmissions are one of the biggest challenges facing healthcare systems worldwide. In the UK, the NHS spends approximately £1.6 billion per year on unplanned readmissions, and research shows that up to 30% of these could be prevented with better discharge planning and follow-up care.

A predictive model like this helps hospitals identify high-risk patients BEFORE they leave, so they can receive enhanced support — medication checks, follow-up appointments, home visits, or referral to community services.

## ❓ Problem This Solves

When a patient is discharged from hospital, doctors must decide:
- Does this patient need extra follow-up support?
- Should we schedule an early outpatient appointment?
- Does this patient need a home visit from a community nurse?

Currently, these decisions are often based on clinical intuition alone. A predictive model provides an objective, data-driven risk score that helps clinicians make better decisions and allocate limited follow-up resources to the patients who need them most.

## 🔬 How It Works

**Step 1: Patient Data (15 Features)**

| Feature | What It Measures |
|---------|-----------------|
| Age | Patient's age at admission |
| Gender | Male (0) or Female (1) |
| Number of diagnoses | How many conditions the patient has |
| Number of medications | How many drugs the patient takes |
| Number of procedures | How many procedures performed during stay |
| Length of stay | Days spent in hospital |
| Number of lab tests | How many blood tests were ordered |
| Previous admissions | How many times admitted in past year |
| Emergency admission | Was this an emergency (1) or planned (0) |
| Charlson score | Comorbidity burden (higher = sicker) |
| Hemoglobin | Blood oxygen-carrying capacity (low = anaemia) |
| Creatinine | Kidney function marker (high = kidney problems) |
| Glucose | Blood sugar level (high = diabetes risk) |
| Sodium | Electrolyte balance |
| Potassium | Electrolyte balance |

**Step 2: Three Models Compared**

| Model | How It Works |
|-------|-------------|
| Logistic Regression | Simple, interpretable — good baseline |
| Random Forest | Uses 200 decision trees voting together |
| Gradient Boosting | Builds trees sequentially, each fixing previous errors |

**Step 3: Results**

| Model | AUC-ROC | F1-Score |
|-------|---------|----------|
| Logistic Regression | 0.78 | 0.71 |
| Random Forest | 0.80 | 0.73 |
| **Gradient Boosting** | **0.82** | **0.75** |

Gradient Boosting performed best with an AUC of 0.82, meaning it correctly ranks a readmitted patient higher than a non-readmitted patient 82% of the time.

**Step 4: Top Risk Factors**

The model identified these as the strongest predictors of readmission:

1. **Number of previous admissions** — patients who have been admitted before are much more likely to be readmitted again
2. **Creatinine level** — elevated creatinine indicates kidney dysfunction, a known readmission risk factor
3. **Charlson comorbidity score** — patients with multiple chronic conditions have higher readmission risk
4. **Age** — older patients face higher readmission rates
5. **Hemoglobin level** — low hemoglobin (anaemia) increases readmission risk

These findings match published clinical research on readmission risk factors.

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| Total patients | 3,000 |
| Readmission rate | ~35% |
| Best model | Gradient Boosting |
| Best AUC-ROC | 0.82 |
| Best F1-Score | 0.75 |
| Features used | 15 |
| Top predictor | Number of previous admissions |

## 🛠️ Tools Used

| What | Tool |
|------|------|
| ML models | Logistic Regression, Random Forest, Gradient Boosting |
| ML framework | scikit-learn |
| Data scaling | StandardScaler |
| Data handling | Pandas, NumPy |
| Plotting | Matplotlib |
| Evaluation | ROC-AUC, F1-Score, Feature Importance |

## 📁 Files In This Project
