# Interview Preparation: College Admission Prediction

## 1. Project Overview

**Problem Statement:** Predict the probability of a student getting admitted to a graduate program based on their academic profile (GRE score, TOEFL score, university rating, SOP, LOR, CGPA, and research experience).

**Objective:** Build a regression model to estimate admission chances, helping students assess their competitiveness for graduate school applications.

**Dataset:** 500 student records with 7 input features and 1 target variable (chance of admit ranging from 0 to 1)

---

## 2. Technical Concepts

### Regression Analysis
- **Continuous Output:** Probability (0-1 range)
- **Supervised Learning:** Learn from historical admission data
- **Feature Correlation:** Identify most influential factors

### Algorithms
- **Linear Regression:** Basic baseline model
- **Polynomial Regression:** Capture non-linear relationships  
- **Ridge/Lasso Regression:** Regularized linear models
- **Random Forest Regressor:** Ensemble method

---

## 3. Libraries & Technologies

- **Pandas:** Data manipulation
- **NumPy:** Numerical operations
- **Matplotlib/Seaborn:** Visualization
- **Scikit-learn:** ML algorithms and evaluation

---

## 4. Mathematical Foundations

### Linear Regression
\[
y = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n + \epsilon
\]

**Ordinary Least Squares (OLS):**
\[
\min_{\beta} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

### Ridge Regression (L2 Regularization)
\[
\min_{\beta} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^{p} \beta_j^2
\]

### Evaluation Metrics

**Mean Squared Error:**
\[
MSE = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

**Root Mean Squared Error:**
\[
RMSE = \sqrt{MSE}
\]

**R² Score (Coefficient of Determination):**
\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
\]

### Pearson Correlation
\[
r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}
\]

---

## 5. Implementation Details

### Dataset Features
| Feature | Description | Range |
|---------|-------------|-------|
| GRE Score | Graduate Record Exam score | 260-340 |
| TOEFL Score | English proficiency test | 0-120 |
| University Rating | Institution ranking | 1-5 |
| SOP | Statement of Purpose strength | 1-5 |
| LOR | Letter of Recommendation strength | 1-5 |
| CGPA | Cumulative GPA | 0-10 |
| Research | Research experience | 0 or 1 |
| **Chance of Admit** | Admission probability | 0-1 |

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('admission_predict.csv')

# EDA
print(df.describe())
print(df.corr())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Feature vs target
sns.scatterplot(x='CGPA', y='Chance of Admit', data=df)

# Prepare data
X = df.drop('Chance of Admit', axis=1)
y = df['Chance of Admit']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Feature importance
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', ascending=False)
print(coefficients)
```

---

## 6. Outcomes & Results

### Typical Performance
- **R² Score:** 0.80-0.82
- **RMSE:** 0.06-0.07
- **MAE:** 0.04-0.05

### Key Insights
- **Most Important Features:** CGPA (0.18), GRE (0.16), TOEFL (0.13)
- **Least Important:** University Rating, Research
- **Strong Correlations:** CGPA and GRE highly correlated with admission

---

## 7. Interview Questions & Answers

**Q1: Why use regression instead of classification?**

**A1:** The target is continuous (probability from 0 to 1), not discrete classes. Regression provides nuanced predictions (e.g., 0.72 vs 0.85 admit chance) rather than binary yes/no.

**Q2: How do you interpret R² score?**

**A2:** R² represents proportion of variance in target explained by features. R²=0.81 means model explains 81% of variance in admission chances. Values range 0-1; higher is better.

**Q3: What is multicollinearity and how does it affect this model?**

**A3:** Multicollinearity occurs when features are highly correlated (e.g., GRE and TOEFL both measure academic ability). Effects:
- Unstable coefficient estimates
- Difficult to determine individual feature importance
- Solution: Use Ridge/Lasso regression or PCA

**Q4: How would you improve this model?**

**A4:** 
1. Feature engineering (GRE×CGPA interaction)
2. Polynomial features for non-linear relationships
3. Ensemble methods (Random Forest, Gradient Boosting)
4. Cross-validation for robust evaluation
5. Collect more data and features (essays, interviews, extracurriculars)

---

## Additional Resources
- Linear Regression: Scikit-learn Documentation
- Graduate Admissions: Kaggle Dataset

