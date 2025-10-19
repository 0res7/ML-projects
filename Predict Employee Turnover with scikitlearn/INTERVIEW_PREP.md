# Interview Preparation: Predict Employee Turnover

## 1. Project Overview

**Problem Statement:** Predict which employees are likely to leave the company based on satisfaction level, performance evaluations, number of projects, average monthly hours, and other HR metrics.

**Objective:** Build classification models to identify at-risk employees, enabling proactive retention strategies and reducing costly turnover.

**Business Impact:** 
- Reduce recruitment/training costs
- Retain top talent
- Improve workforce planning
- Target retention interventions

---

## 2. Technical Concepts

### Binary Classification
- **Target:** 0 (Stay) vs 1 (Leave)
- **Predictive Analytics:** Identify patterns leading to attrition
- **Feature Engineering:** Create meaningful HR indicators

### Algorithms
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting (XGBoost)**
- **Neural Networks (optional)**

---

## 3. Mathematical Foundations

### Logistic Regression
\[
P(\text{leave}|x) = \frac{1}{1 + e^{-(w^Tx + b)}}
\]

### Gini Impurity (Decision Trees)
\[
G = 1 - \sum_{i=1}^{C} p_i^2
\]

### Gradient Boosting
Sequential ensemble:
\[
F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)
\]
where \(\nu\) is learning rate, \(h_m\) is weak learner.

### Evaluation Metrics
- **Precision:** Of predicted leavers, how many actually left?
- **Recall:** Of actual leavers, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Discrimination ability

---

## 4. Implementation Details

### Typical Features
| Feature | Description | Type |
|---------|-------------|------|
| satisfaction_level | Self-reported satisfaction (0-1) | Numeric |
| last_evaluation | Performance score (0-1) | Numeric |
| number_project | Number of projects handled | Numeric |
| average_monthly_hours | Hours worked per month | Numeric |
| time_spend_company | Years at company | Numeric |
| work_accident | Had workplace accident | Binary |
| promotion_last_5years | Promoted in last 5 years | Binary |
| department | Department name | Categorical |
| salary | Salary level (low/medium/high) | Categorical |
| **left** | Employee left company | **Binary (Target)** |

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('HR_data.csv')

# EDA
print(df.head())
print(df.info())
print(df['left'].value_counts())  # Check imbalance

# Visualize turnover by department
pd.crosstab(df['department'], df['left']).plot(kind='bar')

# Satisfaction vs turnover
df[df['left']==1]['satisfaction_level'].hist(alpha=0.5, label='Left')
df[df['left']==0]['satisfaction_level'].hist(alpha=0.5, label='Stayed')
plt.legend()

# Feature engineering
df['overworked'] = (df['average_monthly_hours'] > 250).astype(int)
df['underutilized'] = (df['number_project'] < 3).astype(int)
df['tenure_satisfaction'] = df['time_spend_company'] * df['satisfaction_level']

# Encode categorical
le = LabelEncoder()
df['salary_encoded'] = le.fit_transform(df['salary'])  # low=0, medium=1, high=2
df = pd.get_dummies(df, columns=['department'], drop_first=True)

# Prepare data
X = df.drop(['left'], axis=1)
y = df['left']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predict
y_pred = rf.predict(X_test_scaled)
y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(10))
```

---

## 5. Outcomes & Results

### Typical Performance
- **Accuracy:** 95-98%
- **Precision:** 90-95%
- **Recall:** 85-90%
- **ROC-AUC:** 0.95-0.98

### Most Important Features
1. **Satisfaction Level:** Strong negative correlation with leaving
2. **Average Monthly Hours:** Overwork increases turnover
3. **Number of Projects:** Both extremes (too few, too many) bad
4. **Last Evaluation:** Low performance or very high (burnout)
5. **Time Spend Company:** Sweet spot around 3-4 years

### Common Turnover Patterns
1. **Dissatisfied Underperformers:** Low satisfaction + low evaluation
2. **Burned Out Stars:** High hours + many projects + low satisfaction
3. **Stagnant Employees:** Long tenure + no promotion
4. **Underpaid Talent:** High performance + low salary

---

## 6. Interview Questions & Answers

**Q1: Why is recall important for employee turnover prediction?**

**A1:** **False Negatives are Costly**
- Missing a flight risk (FN) → Employee leaves → Replacement cost ($50K-200K)
- False alarm (FP) → Unnecessary retention effort → Lower cost

**Strategy:** Optimize for high recall
- Cast wide net for at-risk employees
- Better to have retention conversations with false positives than miss true leavers

**Q2: How do you handle the class imbalance?**

**A2:** Turnover typically 10-20% (imbalanced)

**Solutions:**
```python
# 1. Stratified sampling
train_test_split(..., stratify=y)

# 2. Class weights
rf = RandomForestClassifier(class_weight='balanced')

# 3. SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# 4. Threshold tuning
y_pred = (y_pred_proba > 0.3).astype(int)  # Lower threshold
```

**Q3: How would you interpret the model to HR stakeholders?**

**A3:** **Actionable Insights:**

"The model identified four key turnover drivers:

1. **Low Satisfaction (<0.4):** Employees scoring below 0.4 on satisfaction surveys are 5× more likely to leave
   - Action: Conduct stay interviews, address concerns

2. **Overwork (>250 hours/month):** Working excessive hours doubles turnover risk
   - Action: Monitor workload, redistribute projects

3. **Stagnation (No promotion in 5 years):** Long-tenured employees without advancement feel stuck
   - Action: Career development plans, internal mobility

4. **Salary:** Low-paid high performers are flight risks
   - Action: Compensation review, market adjustments

The model provides a risk score for each employee, enabling proactive retention efforts."

**Q4: What are the ethical considerations?**

**A4:**

**Concerns:**
1. **Privacy:** Using personal data for predictions
2. **Self-Fulfilling Prophecy:** Treating predicted leavers differently
3. **Bias:** Model may discriminate (age, gender, department)
4. **Transparency:** Employees deserve to know they're being monitored

**Best Practices:**
1. Anonymize data for analysis
2. Use predictions for systemic improvements, not individual targeting
3. Regular bias audits
4. Transparent HR policies
5. Employee consent for data usage

**Q5: How would you deploy this model in production?**

**A5:**

**Deployment Architecture:**
```python
# 1. Save trained model
import joblib
joblib.dump(rf, 'turnover_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 2. Create prediction API
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load('turnover_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = preprocess(data)
    risk_score = model.predict_proba(features)[0][1]
    return jsonify({'turnover_risk': float(risk_score)})

# 3. Batch scoring
# Monthly: Score all employees, generate risk reports

# 4. Dashboard
# Power BI / Tableau dashboard showing:
# - High-risk employees by department
# - Turnover trends
# - Feature contributions per employee
```

**Monitoring:**
- Track prediction accuracy
- Retrain quarterly with new data
- Monitor for concept drift
- A/B test retention interventions

---

## Additional Resources

**Research:**
- "Why Good Employees Leave" - Harvard Business Review
- Predictive analytics in HR: Deloitte insights

**Tools:**
- SHAP values for explainable AI
- Survival analysis for time-to-turnover

