# Interview Preparation: Heart Disease Prediction

## 1. Project Overview

**Problem Statement:** Predict the presence of heart disease in patients based on clinical parameters using machine learning classification algorithms.

**Objective:** Build a binary classification model to identify patients at risk of heart disease using medical attributes like age, blood pressure, cholesterol, and ECG results.

**Medical Significance:** Early detection enables preventive measures, lifestyle changes, and timely medical intervention, potentially saving lives.

**Dataset:** Cleveland Heart Disease dataset (303 patients, 14 attributes)

---

## 2. Technical Concepts

### Binary Classification
- **Target Variable:** 0 (No disease) vs 1 (Disease present)
- **Supervised Learning:** Learn from labeled patient data
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

### Machine Learning Algorithms
- **Logistic Regression:** Linear decision boundary
- **Random Forest:** Ensemble of decision trees
- **Support Vector Machine (SVM):** Maximum margin classifier
- **K-Nearest Neighbors (KNN):** Instance-based learning

---

## 3. Libraries & Technologies

### Core Libraries
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **Matplotlib/Seaborn:** Data visualization
- **Scikit-learn:** Machine learning algorithms
  - `train_test_split`: Split data
  - `StandardScaler`: Feature scaling
  - `LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`
  - `accuracy_score`, `confusion_matrix`, `classification_report`, `roc_auc_score`

---

## 4. Code Architecture & Design Patterns

### File Structure
```
Heart Disease Prediction [END 2 END]/
├── Heart Disease Prediction.ipynb    # Main analysis notebook
├── heart.csv                          # Dataset
└── README.md
```

### ML Pipeline
```
Data Loading → EDA → Data Preprocessing → 
Feature Selection → Train-Test Split → 
Model Training → Evaluation → Comparison
```

---

## 5. Mathematical Foundations

### Logistic Regression
Sigmoid function:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

where \(z = w^T x + b\)

**Log Loss (Binary Cross-Entropy):**
\[
L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]
\]

### Random Forest
Combines multiple decision trees:
\[
\hat{y} = \text{mode}(\{h_1(x), h_2(x), ..., h_T(x)\})
\]

**Gini Impurity:**
\[
G = 1 - \sum_{i=1}^{C} p_i^2
\]
where \(p_i\) is probability of class \(i\).

### Support Vector Machine
Find hyperplane maximizing margin:
\[
\text{minimize } \frac{1}{2}||w||^2 + C\sum_{i=1}^{N} \xi_i
\]
subject to \(y_i(w^T x_i + b) \geq 1 - \xi_i\)

### Evaluation Metrics

**Accuracy:**
\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

**Precision:**
\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Recall (Sensitivity):**
\[
\text{Recall} = \frac{TP}{TP + FN}
\]

**F1-Score:**
\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**ROC-AUC:** Area under Receiver Operating Characteristic curve

---

## 6. Implementation Details

### Dataset Features

| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numeric |
| sex | Gender (1=male, 0=female) | Binary |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numeric |
| chol | Serum cholesterol (mg/dl) | Numeric |
| fbs | Fasting blood sugar > 120 mg/dl (1=true) | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numeric |
| exang | Exercise induced angina (1=yes) | Binary |
| oldpeak | ST depression induced by exercise | Numeric |
| slope | Slope of peak exercise ST segment (0-2) | Categorical |
| ca | Number of major vessels colored (0-3) | Numeric |
| thal | Thalassemia (0-3) | Categorical |
| **target** | Heart disease (0=no, 1=yes) | **Binary** |

### Complete Workflow

**1. Data Loading & Exploration**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('heart.csv')

# Explore
print(df.shape)        # (303, 14)
print(df.info())       # Check data types
print(df.isnull().sum())  # No missing values
print(df.describe())   # Statistics
```

**2. Data Visualization**
```python
# Target distribution
sns.countplot(df['target'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# Age vs target
sns.boxplot(x='target', y='age', data=df)

# Feature distributions by target
df.hist(figsize=(12, 10), bins=20)
```

**3. Data Preprocessing**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**4. Model Training**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Train all models
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully")
```

**5. Model Evaluation**
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

results = {}

for name, model in trained_models.items():
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'roc_auc': roc_auc
    }
    
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
```

**6. Model Comparison**
```python
# Compare accuracies
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

print(comparison_df.sort_values('Accuracy', ascending=False))

# Visualize
plt.figure(figsize=(10, 6))
comparison_df.set_index('Model')[['Accuracy', 'ROC-AUC']].plot(kind='bar')
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
```

---

## 7. Coding Concepts

### Stratified Splitting
```python
# Ensures train and test have same class distribution
train_test_split(..., stratify=y)
```

### Feature Scaling Importance
```python
# SVM and KNN sensitive to scale
# Random Forest not sensitive (tree-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
X_test_scaled = scaler.transform(X_test)        # Transform test with train stats
```

### Model Dictionary Pattern
```python
# Organize multiple models elegantly
models = {'name': ModelClass(), ...}
for name, model in models.items():
    model.fit(X_train, y_train)
```

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **Binary Classification** | Predicting one of two classes (0 or 1) |
| **Supervised Learning** | Learning from labeled data |
| **Feature Scaling** | Normalizing features to similar ranges |
| **Stratified Split** | Maintaining class proportions in train/test |
| **Cross-Validation** | Evaluating model on multiple train/test splits |
| **Confusion Matrix** | Table showing TP, TN, FP, FN |
| **ROC Curve** | Plot of True Positive Rate vs False Positive Rate |
| **AUC** | Area Under Curve (ROC), measures discriminative ability |
| **Angina** | Chest pain due to reduced blood flow to heart |
| **Thalassemia** | Inherited blood disorder affecting hemoglobin |
| **ST Depression** | ECG abnormality indicating ischemia |

---

## 9. Outcomes & Results

### Typical Performance
- **Logistic Regression:** 85-87% accuracy
- **Random Forest:** 87-90% accuracy (often best)
- **SVM:** 85-88% accuracy
- **KNN:** 82-85% accuracy

### Key Insights
- **Most Important Features:** cp (chest pain), thalach (max heart rate), oldpeak (ST depression)
- **Age & Gender:** Males and older patients at higher risk
- **Cholesterol:** Surprisingly weak predictor (contradicts medical intuition)

---

## 10. Interview Questions & Answers

### Q1: Why use StandardScaler for this dataset?

**A1:** 
1. **Different Scales:** age (29-77), chol (126-564), thalach (71-202)
2. **Distance-Based Algorithms:** SVM and KNN compute distances, need uniform scales
3. **Gradient Descent:** Logistic Regression converges faster with scaled features

**Not Needed For:** Random Forest (scale-invariant)

### Q2: What's the difference between accuracy and ROC-AUC?

**A2:**
- **Accuracy:** Proportion of correct predictions (threshold-dependent)
- **ROC-AUC:** Measures discriminative ability across all thresholds (threshold-independent)

**Example:**
```
Imbalanced dataset: 95% class 0, 5% class 1
Model always predicts 0: Accuracy = 95%, but useless!
ROC-AUC would be 0.5 (random guessing)
```

**For Medical:** ROC-AUC better for imbalanced classes

### Q3: Why is recall important for heart disease prediction?

**A3:** **False Negatives are Dangerous**
- Miss diagnosing a sick patient (FN) → No treatment → Fatal
- False positive (FP) → Extra tests → Inconvenience but safe

**Optimize for High Recall:**
```python
# Lower threshold to increase recall
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_pred_proba > 0.3).astype(int)  # Instead of 0.5
```

### Q4: How would you handle class imbalance?

**A4:**
```python
# 1. SMOTE (Synthetic Minority Over-sampling)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# 2. Class Weights
model = LogisticRegression(class_weight='balanced')

# 3. Stratified Sampling
train_test_split(..., stratify=y)
```

### Q5: Explain feature importance in Random Forest.

**A5:**
```python
# Get feature importances
rf = RandomForestClassifier(...)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize
plt.barh(importances['feature'], importances['importance'])
```

**Interpretation:** Higher importance = more useful for splitting

**Top Features (typical):**
1. cp (chest pain type)
2. thalach (max heart rate)
3. oldpeak (ST depression)
4. ca (number of vessels)

---

## Additional Resources

**Datasets:**
- Cleveland Heart Disease Database (UCI ML Repository)
- Framingham Heart Study
- MIMIC-III Clinical Database

**Papers:**
- Detrano et al. (1989): "International Application of Heart Disease"
- Mohan et al. (2019): "Effective Heart Disease Prediction using Hybrid Machine Learning"

