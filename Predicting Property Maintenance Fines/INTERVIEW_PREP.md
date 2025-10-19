# Interview Preparation: Predicting Property Maintenance Fines

## 1. Project Overview

**Problem Statement:** Predict whether property maintenance violations will result in compliance (payment of fines) based on violation characteristics, property information, and historical patterns.

**Objective:** Build classification models to predict compliance likelihood, helping municipalities optimize enforcement resources and revenue collection.

**Use Case:** Municipal code enforcement, resource allocation, collection prioritization

---

## 2. Technical Concepts

### Binary Classification
- **Target:** 0 (Non-compliant) vs 1 (Compliant/Paid)
- **Imbalanced Classes:** Most violations not paid
- **Time-Series Aspect:** Historical compliance patterns

### Algorithms
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting (XGBoost, LightGBM)**
- **Neural Networks**

---

## 3. Mathematical Foundations

### Logistic Regression
\[
P(y=1|x) = \frac{1}{1 + e^{-(w^Tx + b)}}
\]

### Log Loss (Binary Cross-Entropy)
\[
L = -\frac{1}{N}\sum[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
\]

### Evaluation Metrics

**ROC-AUC:** Most appropriate for imbalanced classes
\[
\text{AUC} = \int_{0}^{1} TPR(FPR^{-1}(x)) dx
\]

**Precision-Recall Curve:** Better than ROC for severe imbalance

---

## 4. Implementation Details

### Typical Features
- **Violation Details:** Type, severity, fine amount
- **Property Info:** Address, zip code, ownership type
- **Temporal:** Month/day of violation, days until hearing
- **Historical:** Previous violations, payment history
- **Spatial:** Neighborhood characteristics, median income

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('property_violations.csv')

# Handle missing values
df['disposition'].fillna('Unknown', inplace=True)
df['fine_amount'].fillna(df['fine_amount'].median(), inplace=True)

# Feature engineering
df['violation_month'] = pd.to_datetime(df['ticket_issued_date']).dt.month
df['days_to_hearing'] = (pd.to_datetime(df['hearing_date']) - 
                          pd.to_datetime(df['ticket_issued_date'])).dt.days

# Encode categoricals
le = LabelEncoder()
df['violation_code_encoded'] = le.fit_transform(df['violation_code'])
df = pd.get_dummies(df, columns=['disposition', 'violation_category'])

# Target variable
# compliance = 1 if fine paid, 0 otherwise
df['compliance'] = df['payment_status'].apply(lambda x: 1 if x == 'Paid in Full' else 0)

# Features and target
X = df[['fine_amount', 'violation_code_encoded', 'days_to_hearing', 
        'violation_month'] + [col for col in df.columns if 'disposition_' in col]]
y = df['compliance']

# Check class imbalance
print(y.value_counts())
print(f"Compliance rate: {y.mean():.2%}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with class weights
xgb = XGBClassifier(
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle imbalance
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_proba = xgb.predict_proba(X_test_scaled)[:, 1]

# Evaluate
auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {auc:.4f}")

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(10))
```

### Handling Severe Class Imbalance
```python
# Typical: 5% compliance, 95% non-compliance

# Method 1: Adjust class weights
scale_pos_weight = count_negative / count_positive

# Method 2: Under-sample majority class
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Method 3: SMOTE (Synthetic Minority Over-sampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# Method 4: Cost-sensitive learning
# Assign higher cost to misclassifying positive class
```

---

## 5. Outcomes & Results

### Typical Performance
- **ROC-AUC:** 0.75-0.82
- **Precision:** 20-30% (at high recall)
- **Recall:** 60-75%

### Most Predictive Features
1. **Fine Amount:** Higher fines less likely to be paid
2. **Violation Type:** Some categories more compliant
3. **Historical Compliance:** Past behavior predicts future
4. **Days to Hearing:** Longer delays reduce compliance
5. **Disposition:** Verdict type affects payment

### Business Value
- **Prioritize Collections:** Focus on high-probability cases
- **Resource Allocation:** Deploy inspectors strategically
- **Policy Insights:** Identify systemic compliance issues
- **Revenue Forecasting:** Predict collection rates

---

## 6. Interview Questions & Answers

**Q1: Why is ROC-AUC preferred over accuracy for this problem?**

**A1:** **Severe Class Imbalance**
```
Compliance rate: 5%
Non-compliance: 95%

Model predicting all "non-compliant":
- Accuracy: 95% (misleading!)
- ROC-AUC: 0.5 (random guessing)
```

**ROC-AUC Benefits:**
- Threshold-independent
- Measures discriminative ability
- Not affected by class imbalance
- Interpretable: 0.5=random, 1.0=perfect

**Alternative:** Precision-Recall AUC for severe imbalance

**Q2: How would you choose the classification threshold?**

**A2:** **Business-Driven Threshold Selection**

```python
# Default threshold: 0.5
# Custom threshold based on business constraints

# Scenario 1: Maximize F1-Score
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_test, (y_pred_proba > t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]

# Scenario 2: Achieve target recall (catch 80% of compliant cases)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
target_recall = 0.8
threshold_80_recall = thresholds[np.argmax(recall >= target_recall)]

# Scenario 3: Cost-based
# Cost of chasing non-compliant: $50
# Value of collecting compliant: $200
# Choose threshold maximizing expected value
```

**Q3: How do you handle temporal leakage in this problem?**

**A3:** **Temporal Leakage: Using Future Information**

**Example Leakage:**
```python
# BAD: Using payment_date to predict compliance
# payment_date is known only AFTER outcome!
```

**Proper Temporal Split:**
```python
# Train on violations before cutoff date
# Test on violations after cutoff date
cutoff_date = '2020-01-01'
train = df[df['ticket_issued_date'] < cutoff_date]
test = df[df['ticket_issued_date'] >= cutoff_date]

# Ensures realistic evaluation
```

**Time-Series Cross-Validation:**
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    # Train and validate
```

**Q4: What ethical considerations exist for this use case?**

**A4:**

**Potential Issues:**
1. **Equity:** Predicting compliance by neighborhood → Discriminatory enforcement
2. **Feedback Loop:** Over-policing certain areas → More violations → Model reinforces bias
3. **Fairness:** Low-income residents may have less ability to pay, not less intent to comply

**Responsible AI:**
1. **Fairness Metrics:** Check for disparate impact by race, income
   ```python
   from aequitas.group import Group
   g = Group()
   xtab, _ = g.get_crosstabs(df)
   # Analyze false positive rates by protected attribute
   ```

2. **Use for Resource Optimization, Not Discrimination:**
   - Help identify systemic issues
   - Allocate assistance programs
   - Not to punish more harshly

3. **Transparency:**
   - Explain model to affected communities
   - Allow challenges to predictions
   - Regular audits

4. **Human Oversight:**
   - Model assists, doesn't replace human judgment
   - Case-by-case review for edge cases

**Q5: How would you explain feature importance to city officials?**

**A5:** **Actionable Insights**

"Our analysis identified key factors affecting compliance:

**1. Fine Amount (35% importance)**
- Finding: Fines >$500 have 40% lower compliance
- Recommendation: Implement payment plans for large fines

**2. Days to Hearing (25% importance)**
- Finding: Delays >60 days reduce compliance by 30%
- Recommendation: Expedite hearing schedule

**3. Violation Type (20% importance)**
- Finding: Safety violations (70% compliance) vs aesthetic (30%)
- Recommendation: Different enforcement strategies per type

**4. Historical Pattern (15% importance)**
- Finding: Repeat violators rarely comply
- Recommendation: Early intervention programs

**5. Notification Method (5% importance)**
- Finding: Certified mail has 2× compliance vs regular mail
- Recommendation: Use certified mail for high-value cases

These insights enable data-driven policy improvements."

---

## Additional Resources

**Papers:**
- Kleinberg et al. (2018): "Human Decisions and Machine Predictions"
- Barocas & Selbst (2016): "Big Data's Disparate Impact"

**Datasets:**
- Detroit Blight Violations (Kaggle)
- NYC Open Data: Building Violations

**Tools:**
- Aequitas: Bias and fairness audit toolkit
- Fairlearn: Fairness-aware machine learning

