# Interview Preparation: Hiring Challenges

## 1. Project Overview

**Problem Statement:** Solve real-world data science hiring challenges involving predictive modeling on tabular datasets, demonstrating ability to handle end-to-end ML workflows under time constraints.

**Objective:** Build production-ready ML models for business problems:
1. **Triglyceride Prediction:** Predict blood triglyceride levels from patient data
2. **Water Potability:** Classify water as potable or not based on quality metrics

**Skills Demonstrated:**
- Data preprocessing and feature engineering
- Model selection and hyperparameter tuning
- Cross-validation and evaluation
- Code organization and documentation
- Working under time/resource constraints

---

## 2. Technical Concepts (Triglyceride Prediction)

### Regression Problem
- **Target:** Triglyceride level (continuous value, mg/dL)
- **Features:** Patient demographics, lifestyle, other blood markers
- **Clinical Importance:** High triglycerides → Heart disease risk

### Algorithms
- **Linear Regression:** Baseline
- **Random Forest Regressor:** Handle non-linear relationships
- **XGBoost:** Gradient boosting for tabular data
- **Neural Networks:** Deep learning approach

---

## 3. Technical Concepts (Water Potability)

### Binary Classification
- **Target:** 0 (Not potable) vs 1 (Potable)
- **Features:** pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, turbidity

### Algorithms
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **SVM**

---

## 4. Mathematical Foundations

### Linear Regression
\[
y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n + \epsilon
\]

### XGBoost (Gradient Boosting)
Sequential ensemble:
\[
F_M(x) = \sum_{m=1}^{M} \nu \cdot f_m(x)
\]
where \(\nu\) is learning rate, \(f_m\) are weak learners.

### Water Quality Metrics

**pH:** Acidity/alkalinity
- Safe range: 6.5-8.5
- Outside range → corrosion or scale

**Hardness:** Calcium/magnesium content
- High → Scale buildup
- Low → Corrosive

**Turbidity:** Cloudiness
- High → Contaminants present

---

## 5. Implementation Details (Triglyceride Prediction)

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# EDA
print(train.describe())
print(train.isnull().sum())
print(train.corr())

# Handle missing values
train.fillna(train.median(), inplace=True)
test.fillna(train.median(), inplace=True)

# Feature engineering
train['bmi'] = train['weight'] / (train['height'] / 100) ** 2
test['bmi'] = test['weight'] / (test['height'] / 100) ** 2

# Separate features and target
X = train.drop(['id', 'triglycerides'], axis=1)
y = train['triglycerides']
X_test = test.drop('id', axis=1)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Try multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    # Train on full training set
    model.fit(X_train_scaled, y_train)
    
    # Validate
    y_pred_val = model.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)
    
    results[name] = {
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'val_rmse': rmse,
        'val_r2': r2
    }
    
    print(f"\n{name}:")
    print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Val RMSE: {rmse:.2f}")
    print(f"  Val R²: {r2:.4f}")

# Choose best model (e.g., XGBoost)
best_model = models['XGBoost']

# Predict on test set
test_predictions = best_model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'triglycerides': test_predictions
})
submission.to_csv('submission.csv', index=False)
```

### Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# XGBoost hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use best model
best_model = grid_search.best_estimator_
```

---

## 6. Implementation Details (Water Potability)

### Workflow
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Handle missing values
for col in train.columns:
    if train[col].isnull().any():
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(train[col].median(), inplace=True)

# Features and target
X = train.drop(['Potability'], axis=1)
y = train['Potability']

# Check class balance
print(y.value_counts())

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest with class weights
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',  # Handle imbalance
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_val)
y_pred_proba = rf.predict_proba(X_val)[:, 1]

print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
print(f"\n{classification_report(y_val, y_pred)}")

# Predict on test
test_predictions = rf.predict(test)

# Submission
submission = pd.DataFrame({
    'id': test.index,
    'Potability': test_predictions
})
submission.to_csv('submission.csv', index=False)
```

---

## 7. Best Practices for Hiring Challenges

### Time Management
```
Hour 1: EDA, understand data
Hour 2: Feature engineering, baseline model
Hour 3: Model optimization, cross-validation
Hour 4: Final predictions, documentation
```

### Code Organization
```python
# main.py
def load_data():
    ...

def preprocess(df):
    ...

def feature_engineering(df):
    ...

def train_model(X, y):
    ...

def predict(model, X):
    ...

if __name__ == "__main__":
    # Clean pipeline
    train, test = load_data()
    train = preprocess(train)
    X, y = feature_engineering(train)
    model = train_model(X, y)
    predictions = predict(model, test)
    save_submission(predictions)
```

### Documentation
```python
"""
Triglyceride Prediction Model

Author: [Your Name]
Date: [Date]

Approach:
- Imputed missing values with median
- Engineered BMI feature
- Tried Linear Regression, Ridge, Random Forest, XGBoost
- XGBoost performed best (R² = 0.82)
- Used 5-fold CV for robust evaluation

To reproduce:
    python main.py
"""
```

---

## 8. Interview Questions & Answers

**Q1: How do you approach a new dataset in a hiring challenge?**

**A1:**

**Systematic Approach:**

**1. Understand Problem (5-10 min):**
- Read description carefully
- Identify task (regression/classification)
- Note evaluation metric
- Check for special requirements

**2. Exploratory Data Analysis (15-20 min):**
```python
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
df.hist(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True)
```

**3. Baseline Model (10-15 min):**
```python
# Quick baseline to beat
from sklearn.dummy import DummyRegressor
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
baseline_score = dummy.score(X_val, y_val)
```

**4. Feature Engineering (20-30 min):**
- Handle missing values
- Encode categoricals
- Create domain-specific features

**5. Model Training (30-40 min):**
- Try multiple algorithms
- Cross-validation
- Choose best performer

**6. Optimization (20-30 min):**
- Hyperparameter tuning
- Feature selection
- Ensemble if time permits

**7. Final Submission (10-15 min):**
- Predict on test set
- Format submission correctly
- Double-check no errors

**Q2: What are common mistakes in hiring challenges?**

**A2:**

**1. Data Leakage:**
```python
# BAD: Fit scaler on all data
scaler.fit(pd.concat([train, test]))

# GOOD: Fit on train only
scaler.fit(train)
test_scaled = scaler.transform(test)
```

**2. Overfitting to Validation Set:**
- Repeatedly tuning on same val set
- Solution: Use cross-validation

**3. Not Checking Submission Format:**
```python
# Ensure column names, data types match template
submission = pd.read_csv('sample_submission.csv')
submission['target'] = predictions
```

**4. Ignoring Missing Values:**
```python
# Check test set for NaNs
print(test.isnull().sum())
```

**5. Not Saving Work:**
```python
# Save model and scaler
import joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

---

## Additional Resources

**Practice Platforms:**
- Kaggle: Competitions and datasets
- DrivenData: Social impact challenges
- AIcrowd: AI challenges
- Zindi: African data science

**Tips:**
- Read past competition winners' solutions
- Maintain code templates for common tasks
- Practice time management

