# Interview Preparation: Linear Regression with Scikit-learn

## 1. Project Overview

**Problem Statement:** Implement linear regression from scratch and using scikit-learn to understand the fundamental concepts of supervised learning, model training, and evaluation.

**Objective:** Master the basics of linear regression including ordinary least squares, gradient descent, evaluation metrics, and regularization techniques.

**Learning Focus:** Foundation for all machine learning - understanding how models learn from data, make predictions, and measure performance.

---

## 2. Technical Concepts

### Linear Regression
- **Supervised Learning:** Learn mapping from features (X) to continuous target (y)
- **Parametric Model:** Assumes linear relationship
- **Least Squares:** Minimize squared errors

### Variants
- **Simple Linear Regression:** One feature
- **Multiple Linear Regression:** Multiple features
- **Polynomial Regression:** Non-linear relationships via polynomial features
- **Ridge Regression:** L2 regularization
- **Lasso Regression:** L1 regularization

---

## 3. Mathematical Foundations

### Linear Model
\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon
\]

**Matrix Form:**
\[
y = X\beta + \epsilon
\]

### Ordinary Least Squares (OLS)
**Objective:** Minimize sum of squared residuals
\[
\min_{\beta} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
\]

**Closed-Form Solution:**
\[
\hat{\beta} = (X^TX)^{-1}X^Ty
\]

### Gradient Descent
**Iterative Optimization:**
\[
\beta := \beta - \alpha \nabla L(\beta)
\]

where \(\alpha\) is learning rate, \(\nabla L\) is gradient of loss.

**Gradient:**
\[
\nabla L = -\frac{2}{N}X^T(y - X\beta)
\]

### Ridge Regression (L2)
\[
\min_{\beta} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^{p}\beta_j^2
\]

**Closed-Form:**
\[
\hat{\beta}_{\text{ridge}} = (X^TX + \alpha I)^{-1}X^Ty
\]

### Lasso Regression (L1)
\[
\min_{\beta} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^{p}|\beta_j|
\]

- **Feature Selection:** Drives some coefficients to exactly zero
- **No Closed-Form:** Requires iterative optimization

### Evaluation Metrics

**Mean Squared Error (MSE):**
\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
\]

**Root Mean Squared Error (RMSE):**
\[
RMSE = \sqrt{MSE}
\]

**Mean Absolute Error (MAE):**
\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
\]

**R² Score:**
\[
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
\]

---

## 4. Implementation Details

### From Scratch Implementation
```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit using Normal Equation: β = (X^T X)^{-1} X^T y
        """
        # Add intercept term (column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
    
    def predict(self, X):
        """Make predictions"""
        return X.dot(self.coef_) + self.intercept_
    
    def score(self, X, y):
        """Compute R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Usage
lr = LinearRegressionScratch()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2 = lr.score(X_test, y_test)
```

### Gradient Descent Implementation
```python
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        # Gradient descent
        for _ in range(self.n_iters):
            # Predictions
            y_pred = X.dot(self.coef_) + self.intercept_
            
            # Compute loss
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # Gradients
            dw = -(2/n_samples) * X.T.dot(y - y_pred)
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # Update parameters
            self.coef_ -= self.lr * dw
            self.intercept_ -= self.lr * db
    
    def predict(self, X):
        return X.dot(self.coef_) + self.intercept_

# Usage
lr_gd = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
lr_gd.fit(X_train, y_train)

# Plot loss curve
plt.plot(lr_gd.losses)
plt.xlabel('Iteration')
plt.ylabel('MSE Loss')
plt.title('Gradient Descent Convergence')
```

### Scikit-learn Implementation
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Simple Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

print(f"Intercept: {lr.intercept_[0]:.4f}")
print(f"Coefficient: {lr.coef_[0][0]:.4f}")

y_pred = lr.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')

# 2. Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

lr_poly = LinearRegression()
lr_poly.fit(X_poly_train, y_train)
y_pred_poly = lr_poly.predict(X_poly_test)

# 3. Ridge Regression (L2 Regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 4. Lasso Regression (L1 Regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

# Compare models
models = {
    'Linear Regression': (lr, y_pred),
    'Polynomial Regression': (lr_poly, y_pred_poly),
    'Ridge': (ridge, y_pred_ridge),
    'Lasso': (lasso, y_pred_lasso)
}

for name, (model, predictions) in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")

# 5. Cross-Validation
scores = cross_val_score(lr, X, y, cv=5, scoring='r2')
print(f"\nCross-Validation R² Scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Feature Scaling
```python
# Standardization (important for Ridge/Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ridge with scaled features
ridge_scaled = Ridge(alpha=1.0)
ridge_scaled.fit(X_train_scaled, y_train)
```

---

## 5. Interview Questions & Answers

**Q1: When would you use Ridge vs Lasso regression?**

**A1:**

**Ridge (L2):**
- **Use When:** Many features, all potentially relevant
- **Effect:** Shrinks coefficients towards zero (but not exactly zero)
- **Advantage:** Handles multicollinearity well
- **Example:** Predicting house prices with 50 features

**Lasso (L1):**
- **Use When:** Feature selection needed
- **Effect:** Drives some coefficients to exactly zero
- **Advantage:** Automatic feature selection
- **Example:** High-dimensional data (p > n)

**Elastic Net:** Combines L1 and L2 (best of both)

**Q2: What assumptions does linear regression make?**

**A2:**

**L.I.N.E. Assumptions:**

1. **Linearity:** Relationship between X and y is linear
2. **Independence:** Observations are independent
3. **Normality:** Residuals are normally distributed
4. **Equal Variance (Homoscedasticity):** Constant variance of residuals

**Check Assumptions:**
```python
# Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')

# Q-Q plot for normality
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
```

**Q3: Why use gradient descent instead of normal equation?**

**A3:**

**Normal Equation:** \(\hat{\beta} = (X^TX)^{-1}X^Ty\)
- **Time Complexity:** O(n³) for matrix inversion
- **Space:** Must compute X^T X (n×n matrix)
- **Good For:** n < 10,000 features

**Gradient Descent:**
- **Time Complexity:** O(knd) where k=iterations, n=samples, d=features
- **Space:** O(d)
- **Good For:** Large datasets, many features

**When n is large:** Gradient descent faster

**Q4: How do you interpret R² score?**

**A4:**

**R² = 0.75**
- Model explains 75% of variance in target
- Remaining 25% unexplained (noise or missing features)

**Range:** 0 to 1 (higher is better)
- R²=0: Model no better than predicting mean
- R²=1: Perfect predictions

**Limitations:**
- Adding features always increases R²
- Use Adjusted R² for model comparison:
  \[
  R^2_{\text{adj}} = 1 - (1-R^2)\frac{n-1}{n-p-1}
  \]

**Q5: What is multicollinearity and how does it affect linear regression?**

**A5:**

**Multicollinearity:** High correlation between features

**Example:**
```
height_cm and height_inches: r = 1.0 (perfect correlation)
```

**Problems:**
1. Unstable coefficient estimates
2. High variance in predictions
3. Difficulty interpreting feature importance

**Detection:**
```python
# Correlation matrix
corr_matrix = X.corr()
high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# VIF > 10 indicates multicollinearity
```

**Solutions:**
1. Remove one of correlated features
2. Use Ridge regression (L2 handles multicollinearity)
3. PCA (combine correlated features)

---

## Additional Resources

**Books:**
- "Introduction to Statistical Learning" by James et al.
- "Pattern Recognition and Machine Learning" by Bishop

**Online:**
- Scikit-learn Documentation: Linear Models
- Andrew Ng's Machine Learning Course (Coursera)

**Practice:**
- Boston Housing Dataset
- California Housing Dataset
- Kaggle: House Prices Competition

