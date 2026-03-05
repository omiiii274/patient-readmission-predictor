
---

### PROJECT 3: `patient-readmission-predictor`

**File: `main.py`**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, classification_report, 
                             roc_curve, confusion_matrix, f1_score)
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')
os.makedirs('images', exist_ok=True)
np.random.seed(42)

# ── Generate Simulated EHR Data ──
print("🏥 Generating simulated EHR patient data...")
n = 3000

age = np.random.normal(65, 15, n).clip(18, 95).astype(int)
gender = np.random.choice([0, 1], n)
num_diagnoses = np.random.poisson(3, n)
num_medications = np.random.poisson(5, n)
num_procedures = np.random.poisson(2, n)
length_of_stay = np.random.exponential(5, n).clip(1, 30).round(1)
num_lab_tests = np.random.poisson(8, n)
num_prev_admissions = np.random.poisson(1.5, n)
emergency_admission = np.random.binomial(1, 0.3, n)
charlson_score = np.random.poisson(2, n)
hemoglobin = np.random.normal(12, 2, n).clip(5, 18).round(1)
creatinine = np.random.exponential(1.2, n).clip(0.5, 8).round(2)
glucose = np.random.normal(120, 40, n).clip(50, 400).round(0)
sodium = np.random.normal(140, 4, n).clip(125, 155).round(0)
potassium = np.random.normal(4.2, 0.6, n).clip(2.5, 6.5).round(1)

# Readmission probability based on features
logit = (-3 + 0.02*age + 0.3*num_prev_admissions + 0.15*charlson_score +
         0.1*emergency_admission + 0.05*length_of_stay - 0.1*hemoglobin +
         0.2*creatinine + 0.003*glucose + np.random.normal(0, 0.5, n))
prob = 1 / (1 + np.exp(-logit))
readmitted = (prob > 0.5).astype(int)

df = pd.DataFrame({
    'age': age, 'gender': gender, 'num_diagnoses': num_diagnoses,
    'num_medications': num_medications, 'num_procedures': num_procedures,
    'length_of_stay': length_of_stay, 'num_lab_tests': num_lab_tests,
    'num_prev_admissions': num_prev_admissions, 'emergency_admission': emergency_admission,
    'charlson_score': charlson_score, 'hemoglobin': hemoglobin, 'creatinine': creatinine,
    'glucose': glucose, 'sodium': sodium, 'potassium': potassium, 'readmitted_30day': readmitted
})
df.to_csv('patient_data.csv', index=False)
print(f"✅ Generated {n} patients | Readmitted: {readmitted.sum()} ({readmitted.mean()*100:.1f}%)")

# ── Train Models ──
features = [c for c in df.columns if c != 'readmitted_30day']
X = df[features]
y = df['readmitted_30day']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
}

results = {}
print("\n📊 Training models...")
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = model.predict(X_test_sc)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'model': model, 'auc': auc, 'f1': f1, 'y_prob': y_prob, 'y_pred': y_pred}
    print(f"  {name}: AUC={auc:.4f}, F1={f1:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
print(f"\n🏆 Best Model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ── Plot 1: ROC Curves ──
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2B7A78', '#E74C3C', '#3498DB']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1)
ax.set_xlabel('False Positive Rate',fontsize=12)
ax.set_ylabel('True Positive Rate',fontsize=12)
ax.set_title('ROC Curves — 30-Day Readmission Prediction',fontsize=14)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('images/roc_curves.png',dpi=150)
print("✅ Saved: images/roc_curves.png")

# ── Plot 2: Feature Importance ──
best_model = results[best_name]['model']
if hasattr(best_model, 'feature_importances_'):
    importance = pd.Series(best_model.feature_importances_, index=features).sort_values()
else:
    importance = pd.Series(np.abs(best_model.coef_[0]), index=features).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
importance.plot(kind='barh', color='#2B7A78', ax=ax)
ax.set_title(f'Feature Importance — {best_name}', fontsize=14)
ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig('images/feature_importance.png', dpi=150)
print("✅ Saved: images/feature_importance.png")

# ── Plot 3: Model Comparison ──
fig, ax = plt.subplots(figsize=(8, 5))
model_names = list(results.keys())
aucs = [results[m]['auc'] for m in model_names]
f1s = [results[m]['f1'] for m in model_names]
x = np.arange(len(model_names))
ax.bar(x-0.2, aucs, 0.35, label='AUC', color='#2B7A78')
ax.bar(x+0.2, f1s, 0.35, label='F1', color='#E74C3C')
ax.set_xticks(x); ax.set_xticklabels(model_names)
ax.set_ylabel('Score'); ax.set_title('Model Comparison')
ax.legend(); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig('images/model_comparison.png', dpi=150)
print("✅ Saved: images/model_comparison.png")

# ── Plot 4: Readmission by Age Group ──
df['age_group'] = pd.cut(df['age'], bins=[0,40,55,65,75,100], labels=['<40','40-54','55-64','65-74','75+'])
age_readmit = df.groupby('age_group')['readmitted_30day'].mean() * 100
fig, ax = plt.subplots(figsize=(8, 5))
age_readmit.plot(kind='bar', color='#2B7A78', ax=ax)
ax.set_ylabel('Readmission Rate (%)'); ax.set_title('30-Day Readmission Rate by Age Group')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('images/readmission_by_age.png', dpi=150)
print("✅ Saved: images/readmission_by_age.png")

print("\n🏁 All done! Check /images/ folder for all plots.")
