# Interview Preparation: Wine Quality Prediction

## 1. Project Overview

**Problem Statement:** Predict wine quality (score 0-10) based on physicochemical properties like acidity, pH, alcohol content, and sulfur dioxide levels.

**Objective:** Build regression or classification models to assess wine quality for quality control, pricing, and certification in the wine industry.

**Dataset:** UCI Wine Quality dataset with ~6,000 samples of red and white wines, 11 physicochemical features, quality scores 3-9

---

## 2. Technical Concepts

### Problem Formulation
- **Regression:** Predict exact quality score (3-9)
- **Classification:** Categorize as low/medium/high quality

### Algorithms
- **Linear Regression / Ridge / Lasso**
- **Random Forest Regressor/Classifier**
- **Gradient Boosting**
- **Support Vector Regression (SVR)**

---

## 3. Mathematical Foundations

### Features & Quality Relationship

**Key Chemical Properties:**
1. **Fixed Acidity:** Tartaric acid content
2. **Volatile Acidity:** Acetic acid (vinegar taste if high)
3. **Citric Acid:** Adds freshness
4. **Residual Sugar:** Sweetness
5. **Chlorides:** Saltiness
6. **Free/Total Sulfur Dioxide:** Preservative
7. **Density:** Related to sugar/alcohol
8. **pH:** Acidity level (3-4 for wine)
9. **Sulphates:** Wine additive
10. **Alcohol:** Ethanol percentage
11. **Quality:** Target (0-10 scale)

### Random Forest Regressor
Ensemble of decision trees:
\[
\hat{y} = \frac{1}{T}\sum_{t=1}^{T} h_t(x)
\]

### Mean Absolute Error
\[
MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|
\]

### R² Score
\[
R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
\]

---

## 4. Implementation Details

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
wine = pd.read_csv('winequality.csv')

# EDA
print(wine.describe())
print(wine['quality'].value_counts())

# Correlation analysis
plt.figure(figsize=(12, 8))
sns.heatmap(wine.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# Most correlated with quality
correlations = wine.corr()['quality'].sort_values(ascending=False)
print(correlations)

# Prepare data
X = wine.drop('quality', axis=1)
y = wine['quality']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regression approach
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)
y_pred = rf_reg.predict(X_test_scaled)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {np.mean(np.abs(y_test - y_pred)):.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_reg.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(importances['feature'], importances['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Wine Quality')

# Classification approach (optional)
# Convert to classes: low (3-5), medium (6), high (7-9)
def categorize_quality(score):
    if score <= 5:
        return 0  # low
    elif score == 6:
        return 1  # medium
    else:
        return 2  # high

y_class = y.apply(categorize_quality)
```

### Handling Imbalanced Quality Scores
```python
# Quality distribution often imbalanced (mostly 5-6)
print(y.value_counts())

# Solution 1: Class weights
from sklearn.utils import class_weight
weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y), y=y
)

# Solution 2: Stratified sampling
train_test_split(..., stratify=y)

# Solution 3: SMOTE for classification
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X_train, y_train)
```

---

## 5. Outcomes & Results

### Typical Performance

**Regression:**
- **R²:** 0.35-0.50
- **RMSE:** 0.6-0.8
- **MAE:** 0.5-0.6

**Classification (3 classes):**
- **Accuracy:** 70-80%
- **F1-Score:** 0.65-0.75

### Most Important Features
1. **Alcohol:** (+) Higher alcohol → Higher quality
2. **Volatile Acidity:** (-) Higher → Lower quality (vinegar taste)
3. **Sulphates:** (+) Moderate levels improve quality
4. **Citric Acid:** (+) Adds freshness
5. **Total Sulfur Dioxide:** (-) Too much affects taste

### Key Insights
- Quality is subjective (human sensory evaluation)
- Most wines rated 5-6 (normal quality)
- Difficult to predict exact scores
- Chemical properties explain only ~40-50% of variance

---

## 6. Interview Questions & Answers

**Q1: Why is R² relatively low (0.35-0.50) for this dataset?**

**A1:** **Quality is Subjective**
- Human taste preferences vary
- Sensory evaluation has inherent noise
- Chemical properties don't capture everything (aroma, appearance)
- R²=0.4 means chemistry explains 40% of quality; remaining 60% from other factors

**Solution:** Accept limitation or collect more features (tannins, aging, storage conditions)

**Q2: Should this be regression or classification?**

**A2:** **Depends on Use Case**

**Regression (Predict Score):**
- Precise quality estimation
- Ranking wines by score
- Use: Pricing, certification

**Classification (Low/Medium/High):**
- Simpler, more interpretable
- Actionable categories
- Use: Quality control (pass/fail), marketing

**Hybrid Approach:** Train regressor, then threshold predictions into classes

**Q3: How do you interpret feature importance from Random Forest?**

**A3:**
```python
# Example output:
# alcohol: 0.25
# volatile_acidity: 0.18
# sulphates: 0.12
```

**Interpretation:**
- Alcohol contributes 25% to model decisions
- More important features used earlier in trees
- Top 3 features account for 55% of predictions

**Caution:**
- Correlated features split importance
- Not causal relationships
- Relative, not absolute importance

**Q4: How would you improve model performance?**

**A4:**

**1. Feature Engineering:**
```python
# Interaction terms
wine['alcohol_acidity'] = wine['alcohol'] * wine['fixed acidity']
wine['sugar_acid_ratio'] = wine['residual sugar'] / wine['total acidity']

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

**2. Advanced Models:**
```python
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

xgb = XGBRegressor(n_estimators=200, learning_rate=0.05)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=4)
```

**3. Ensemble Methods:**
```python
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('rf', RandomForestRegressor(100)),
    ('gb', GradientBoostingRegressor(100)),
    ('xgb', XGBRegressor(100))
])
```

**4. Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=10, scoring='r2')
print(f"R²: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

**Q5: Explain the role of alcohol in wine quality.**

**A5:**

**Positive Correlation with Quality:**
- Higher alcohol → Better quality (typically)
- Alcohol contributes to body, mouthfeel
- Extracted from sugars during fermentation

**Chemical Relationship:**
```
Sugar (grapes) → Fermentation → Alcohol + CO₂
More ripe grapes → More sugar → More alcohol → Better wine
```

**Optimal Range:**
- Table wine: 11-14% alcohol
- Too low (<10%): Thin, watery
- Too high (>15%): Overpowering, hot

**Feature Engineering:**
```python
# Alcohol relative to sugar
wine['alcohol_sugar_ratio'] = wine['alcohol'] / (wine['residual sugar'] + 1)
```

---

## Additional Resources

**Dataset:**
- UCI Machine Learning Repository: Wine Quality Data Set
- P. Cortez et al. (2009): "Modeling wine preferences by data mining from physicochemical properties"

**Wine Chemistry:**
- "The Science of Wine: From Vine to Glass" by Jamie Goode
- "Postmodern Winemaking" by Clark Smith

**Machine Learning:**
- Regression analysis techniques
- Feature importance interpretation

