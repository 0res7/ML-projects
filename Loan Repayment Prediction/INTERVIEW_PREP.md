# Interview Preparation: Loan Repayment Prediction

## 1. Project Overview

**Problem Statement:** Predict whether a borrower will repay a loan or default based on financial and demographic features, helping lenders make informed credit decisions.

**Objective:** Build binary classification models to assess credit risk and minimize loan defaults while maximizing approval rates for creditworthy borrowers.

**Business Impact:** Reduces financial losses from defaults, improves loan approval efficiency, enables data-driven lending decisions.

---

## 2. Technical Concepts

### Binary Classification
- **Target:** 0 (Repay) vs 1 (Default)
- **Imbalanced Classes:** Typically more repayers than defaulters
- **Cost-Sensitive:** False negatives (approve bad loan) costly

### Algorithms
- **Logistic Regression:** Probabilistic linear classifier
- **Decision Tree:** Rule-based classification
- **Random Forest:** Ensemble method
- **Gradient Boosting:** Sequential ensemble
- **XGBoost:** Optimized gradient boosting

---

## 3. Mathematical Foundations

### Logistic Regression
**Sigmoid Function:**
\[
P(y=1|x) = \frac{1}{1 + e^{-(w^Tx + b)}}
\]

**Log Loss:**
\[
L = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
\]

### Confusion Matrix
\[
\begin{bmatrix}
TN & FP \\
FN & TP
\end{bmatrix}
\]

### Evaluation Metrics

**Precision (Positive Predictive Value):**
\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Recall (Sensitivity, True Positive Rate):**
\[
\text{Recall} = \frac{TP}{TP + FN}
\]

**F1-Score:**
\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**ROC-AUC:** Area under Receiver Operating Characteristic curve

---

## 4. Implementation Details

### Typical Features
- **Demographic:** Age, employment status, education
- **Financial:** Income, debt-to-income ratio, credit history
- **Loan-specific:** Amount, term, purpose, interest rate

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv('loan_data.csv')

# Handle missing values
df.fillna(df.median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['purpose'] = le.fit_transform(df['purpose'])

# Feature engineering
df['debt_to_income'] = df['monthly_debt'] / df['monthly_income']
df['loan_to_income'] = df['loan_amount'] / df['annual_income']

# Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']  # 0=repay, 1=default

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
```

### Handling Class Imbalance
```python
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

# Method 1: SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Method 2: Class weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
model = LogisticRegression(class_weight='balanced')

# Method 3: Threshold tuning
# Lower threshold to increase recall (catch more defaults)
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_pred_proba > 0.3).astype(int)  # Instead of 0.5
```

---

## 5. Outcomes & Results

### Typical Performance
- **Accuracy:** 75-85%
- **Precision:** 70-80%
- **Recall:** 65-75%
- **ROC-AUC:** 0.75-0.85

### Feature Importance
1. Credit history/score
2. Debt-to-income ratio
3. Loan amount
4. Employment length
5. Annual income

---

## 6. Interview Questions & Answers

**Q1: Why is recall more important than precision for loan default prediction?**

**A1:** **False Negative (FN) is Costly**
- FN: Approve loan for defaulter → Financial loss
- FP: Reject loan for good borrower → Missed opportunity (less costly)

**Optimize for High Recall:**
- Catch more potential defaulters
- Accept some false alarms
- Use threshold tuning: lower from 0.5 to 0.3-0.4

**Q2: How do you handle class imbalance?**

**A2:**
1. **SMOTE:** Synthetic minority over-sampling
2. **Class Weights:** Penalize minority class errors more
3. **Threshold Adjustment:** Lower threshold for minority class
4. **Ensemble Methods:** Often naturally handle imbalance better
5. **Evaluation Metrics:** Use F1-score, ROC-AUC instead of accuracy

**Q3: What is the difference between ROC-AUC and accuracy?**

**A3:**
- **Accuracy:** Overall correctness, but misleading with imbalance
  - Example: 95% repay, 5% default. Predict all "repay" → 95% accuracy but useless!
- **ROC-AUC:** Threshold-independent, measures discriminative ability
  - AUC=0.5: Random guessing
  - AUC=1.0: Perfect discrimination
  - Better for imbalanced classes

**Q4: How would you explain the model to non-technical stakeholders?**

**A4:** "The model analyzes historical loan data to identify patterns that distinguish borrowers who repaid from those who defaulted. It assigns a risk score (0-100) to new applicants. Higher scores indicate higher default risk. We can set a cutoff (e.g., 70) where scores above trigger additional review or rejection."

**Q5: What are the ethical considerations?**

**A5:**
1. **Fairness:** Model shouldn't discriminate based on protected attributes (race, gender)
2. **Transparency:** Explainable predictions (use SHAP values)
3. **Bias:** Historical data may reflect past discrimination
4. **Regulations:** Comply with Equal Credit Opportunity Act, Fair Lending laws
5. **Monitoring:** Regular audits for disparate impact

---

## Additional Resources
- Credit Risk Modeling: "Credit Risk Analytics" by Bart Baesens
- Fair Lending: Consumer Financial Protection Bureau guidelines
- SMOTE: Chawla et al. (2002)

