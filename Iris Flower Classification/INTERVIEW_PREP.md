# Interview Preparation: Iris Flower Classification

## 1. Project Overview

**Problem Statement:** Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on four physical measurements: sepal length, sepal width, petal length, and petal width.

**Objective:** Build multi-class classification models using classic machine learning algorithms and compare their performance on this foundational dataset.

**Historical Significance:** Introduced by statistician Ronald Fisher in 1936, the Iris dataset is one of the most famous datasets in machine learning and statistics, often used as a "Hello World" for ML practitioners.

**Dataset:** 150 samples (50 per species), 4 features, perfectly balanced classes

---

## 2. Technical Concepts

### Multi-Class Classification
- **Three Classes:** Setosa, Versicolor, Virginica
- **Supervised Learning:** Learn from labeled data
- **Linear Separability:** Setosa is linearly separable; Versicolor and Virginica overlap slightly

### Machine Learning Algorithms
- **K-Nearest Neighbors (KNN):** Instance-based learning
- **Support Vector Machine (SVM):** Maximum margin classifier with kernels
- **Decision Tree:** Recursive binary splitting
- **Logistic Regression:** Linear probabilistic model
- **Naive Bayes:** Probabilistic classifier based on Bayes' theorem
- **Random Forest:** Ensemble of decision trees

---

## 3. Libraries & Technologies

### Core Libraries
- **Pandas:** Data manipulation
- **NumPy:** Numerical operations
- **Matplotlib/Seaborn:** Data visualization
- **Scikit-learn:** Machine learning algorithms
  - `train_test_split`: Data splitting
  - `StandardScaler`: Feature scaling
  - `KNeighborsClassifier`, `SVC`, `DecisionTreeClassifier`, `LogisticRegression`, `GaussianNB`, `RandomForestClassifier`
  - `accuracy_score`, `confusion_matrix`, `classification_report`

---

## 4. Code Architecture & Design Patterns

### File Structure
```
Iris Flower Classification/
├── iris.ipynb                  # Main analysis
├── SVM Iris.ipynb              # SVM kernel comparison
├── KNN on Iris Dataset/
│   └── iris_Flower_Classification_using_KNN.ipynb
└── INTERVIEW_PREP.md
```

### ML Pipeline
```
Data Loading → EDA → Visualization → Feature Scaling → 
Train-Test Split → Model Training → Evaluation → Comparison
```

---

## 5. Mathematical Foundations

### K-Nearest Neighbors
**Distance Metric (Euclidean):**
\[
d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}
\]

**Classification:**
\[
\hat{y} = \text{mode}(\{y_1, y_2, ..., y_k\})
\]

where \(y_1, ..., y_k\) are labels of k nearest neighbors.

### Support Vector Machine
**Objective (Soft-margin):**
\[
\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{N}\xi_i
\]

subject to: \(y_i(w^Tx_i + b) \geq 1 - \xi_i\), \(\xi_i \geq 0\)

**Kernel Trick:**
\[
K(x, x') = \phi(x)^T\phi(x')
\]

**Common Kernels:**
- Linear: \(K(x, x') = x^Tx'\)
- Polynomial: \(K(x, x') = (x^Tx' + c)^d\)
- RBF: \(K(x, x') = \exp(-\gamma||x-x'||^2)\)

### Decision Tree (Gini Impurity)
\[
G = 1 - \sum_{i=1}^{C} p_i^2
\]

where \(p_i\) is the probability of class \(i\).

**Information Gain:**
\[
IG = G_{\text{parent}} - \sum_{j} \frac{N_j}{N} G_j
\]

### Logistic Regression
**Softmax for Multi-class:**
\[
P(y=k|x) = \frac{e^{w_k^Tx}}{\sum_{j=1}^{K} e^{w_j^Tx}}
\]

**Cross-Entropy Loss:**
\[
L = -\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
\]

### Naive Bayes
**Bayes' Theorem:**
\[
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
\]

**Gaussian Naive Bayes:**
\[
P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i-\mu_y)^2}{2\sigma_y^2}\right)
\]

### Evaluation Metrics
**Accuracy:**
\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

**Precision (per class):**
\[
\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}
\]

**Recall (per class):**
\[
\text{Recall}_k = \frac{TP_k}{TP_k + FN_k}
\]

**F1-Score:**
\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

---

## 6. Implementation Details

### Dataset Features
| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| sepal_length | Length of sepal | 4.3-7.9 | cm |
| sepal_width | Width of sepal | 2.0-4.4 | cm |
| petal_length | Length of petal | 1.0-6.9 | cm |
| petal_width | Width of petal | 0.1-2.5 | cm |
| **species** | Target class | Setosa, Versicolor, Virginica | categorical |

### Complete Workflow

**1. Data Loading & EDA**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Or load from CSV
df = pd.read_csv('iris.csv')

# Explore
print(df.shape)        # (150, 5)
print(df.info())
print(df.describe())
print(df['species'].value_counts())  # 50, 50, 50 (balanced)
```

**2. Data Visualization**
```python
# Pairplot
sns.pairplot(df, hue='species')

# Box plots
df.boxplot(by='species', figsize=(12, 6))

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Feature distributions
df.hist(figsize=(10, 8))
```

**3. Data Preprocessing**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**4. Model Training & Evaluation**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate
results = {}
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'accuracy': acc,
        'confusion_matrix': cm
    }
    
    print(f"\n{name}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
```

**5. SVM Kernel Comparison**
```python
import matplotlib.pyplot as plt
from sklearn.svm import SVC

kernels = ['linear', 'poly', 'rbf']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, gamma='auto')
    svm.fit(X_train_scaled[:, :2], y_train)  # Use only 2 features for visualization
    
    # Plot decision boundary
    # ... (decision boundary plotting code)
    
    axes[idx].set_title(f'SVM with {kernel} kernel')
```

**6. Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**7. Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

# KNN hyperparameter tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## 7. Coding Concepts

### Stratified Splitting
```python
# Ensures equal class distribution in train and test
train_test_split(..., stratify=y)
```

### Feature Scaling Importance
```python
# Distance-based algorithms (KNN, SVM) need scaling
# Tree-based algorithms (Decision Tree, Random Forest) don't

# Fit scaler on training data only!
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

### Model Comparison Pattern
```python
# Organize multiple models in dictionary
models = {'name': ModelClass(), ...}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = evaluate(model, X_test, y_test)

# Compare
comparison_df = pd.DataFrame(results).T
print(comparison_df.sort_values('accuracy', ascending=False))
```

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **Sepal** | Outer part of flower, typically green, protects bud |
| **Petal** | Colored part of flower, attracts pollinators |
| **Multi-class Classification** | Predicting one of three or more classes |
| **Stratified Sampling** | Maintaining class proportions in splits |
| **Kernel Trick** | Implicitly mapping data to higher dimensions |
| **Hyperplane** | Decision boundary in high-dimensional space |
| **Support Vectors** | Data points closest to decision boundary |
| **Gini Impurity** | Measure of node impurity in decision trees |
| **Information Gain** | Reduction in entropy after split |
| **Cross-Validation** | Evaluating model on multiple train/test splits |
| **Overfitting** | Model learns training noise, poor generalization |
| **Underfitting** | Model too simple, misses patterns |
| **Bias-Variance Tradeoff** | Balance between model simplicity and flexibility |

---

## 9. Outcomes & Results

### Typical Performance (on test set)
- **SVM (RBF):** 98-100% accuracy
- **SVM (Linear):** 96-98% accuracy
- **Logistic Regression:** 95-98% accuracy
- **Naive Bayes:** 95-97% accuracy
- **Random Forest:** 95-97% accuracy
- **Decision Tree:** 93-96% accuracy
- **KNN:** 93-95% accuracy (k=3-5)

### Key Insights
1. **Setosa Perfectly Separable:** All models correctly classify Setosa
2. **Versicolor-Virginica Overlap:** Slight confusion between these two species
3. **Small Dataset:** 150 samples total, good for learning but limited for deep learning
4. **High Performance:** Most models achieve >95% accuracy (well-separated classes)

### Feature Importance (from Decision Tree/Random Forest)
1. **Petal Length:** Most discriminative feature
2. **Petal Width:** Second most important
3. **Sepal Length:** Moderate importance
4. **Sepal Width:** Least important

---

## 10. Interview Questions & Answers

### Conceptual Questions

**Q1: What is the difference between supervised and unsupervised learning? Which category does this project fall into?**

**A1:** 

**Supervised Learning:**
- Training data has labels (input-output pairs)
- Goal: Learn mapping \(f: X \rightarrow Y\)
- Examples: Classification, Regression
- **This project:** Supervised (species labels provided)

**Unsupervised Learning:**
- Training data has no labels (only inputs)
- Goal: Find hidden structure/patterns
- Examples: Clustering, Dimensionality Reduction

**Semi-Supervised:**
- Mix of labeled and unlabeled data
- Example: Few labeled examples + many unlabeled

**Reinforcement Learning:**
- Agent learns by interacting with environment
- Receives rewards/penalties
- Example: Game playing, robotics

**Q2: Explain the bias-variance tradeoff. How does it apply to the models used in this project?**

**A2:**

**Bias:** Error from oversimplifying problem
- High bias → Underfitting
- Model misses important patterns
- Example: Linear model for non-linear data

**Variance:** Sensitivity to training data fluctuations
- High variance → Overfitting
- Model learns noise in training data
- Example: Deep decision tree

**Tradeoff:**
\[
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\]

**For Iris Models:**

**High Bias (Underfit):**
- Linear SVM on non-linearly separable Versicolor/Virginica
- Solution: Use RBF kernel

**High Variance (Overfit):**
- Decision Tree with no depth limit
- Learns specific training examples
- Solution: Prune tree or use Random Forest

**Good Balance:**
- SVM with RBF kernel (C parameter tuned)
- Random Forest (ensemble reduces variance)
- Logistic Regression with regularization

**Q3: Why is feature scaling important for this dataset?**

**A3:**

**Feature Ranges:**
```
sepal_length: [4.3, 7.9] cm
sepal_width:  [2.0, 4.4] cm
petal_length: [1.0, 6.9] cm
petal_width:  [0.1, 2.5] cm
```

**Problems Without Scaling:**

**1. Distance-Based Algorithms (KNN, SVM):**
- Euclidean distance dominated by large-scale features
- Example:
  ```
  d = sqrt((Δsepal_length)² + (Δpetal_width)²)
  Without scaling: sepal_length (range ~3.6) dominates petal_width (range ~2.4)
  ```

**2. Gradient Descent (Logistic Regression):**
- Features on different scales → elongated contours
- Slow convergence, zig-zagging

**StandardScaler:**
\[
z = \frac{x - \mu}{\sigma}
\]
Transforms to mean=0, std=1

**Not Needed For:**
- Decision Trees: Split on thresholds, scale-invariant
- Random Forest: Ensemble of trees
- Naive Bayes: Works with probabilities

---

### Technical Questions

**Q4: Explain how K-Nearest Neighbors (KNN) works.**

**A4:**

**Algorithm:**
1. **Choose k:** Number of neighbors (hyperparameter)
2. **Compute Distances:** Calculate distance to all training points
3. **Find k Nearest:** Select k closest points
4. **Vote:** Majority class among k neighbors

**Distance Metrics:**
- **Euclidean:** \(d = \sqrt{\sum(x_i - y_i)^2}\)
- **Manhattan:** \(d = \sum|x_i - y_i|\)
- **Minkowski:** \(d = (\sum|x_i - y_i|^p)^{1/p}\)

**Choosing k:**
- **k=1:** Very flexible, high variance (overfitting)
- **k=large:** Smooth boundary, high bias (underfitting)
- **Optimal:** Cross-validation to find best k

**Example (Iris):**
```python
# New flower: sepal_length=5.0, sepal_width=3.0, petal_length=1.5, petal_width=0.3

# Find 3 nearest neighbors:
# Neighbor 1: Setosa (distance 0.5)
# Neighbor 2: Setosa (distance 0.6)
# Neighbor 3: Setosa (distance 0.7)

# Prediction: Setosa (3/3 votes)
```

**Computational Complexity:**
- Training: O(1) (just store data)
- Prediction: O(Nd) (N samples, d dimensions)
- **Slow for large datasets!**

**Q5: What is the difference between hard-margin and soft-margin SVM?**

**A5:**

**Hard-Margin SVM:**
```python
minimize: (1/2)||w||²
subject to: y_i(w^T x_i + b) ≥ 1 for all i
```

**Requirements:**
- Data must be linearly separable
- No misclassifications allowed
- **Problem:** Rarely satisfied in real data
- **Iris:** Setosa vs others (separable), but Versicolor vs Virginica (not perfectly separable)

**Soft-Margin SVM:**
```python
minimize: (1/2)||w||² + C Σξ_i
subject to: y_i(w^T x_i + b) ≥ 1 - ξ_i, ξ_i ≥ 0
```

**Slack Variables (ξ_i):**
- Allow some points to violate margin
- Penalty for violations

**C Parameter:**
- **Small C:** Wide margin, more violations (high bias)
- **Large C:** Narrow margin, fewer violations (high variance)

**Example:**
```python
# C=0.1: Tolerant, smooth boundary
svm = SVC(kernel='linear', C=0.1)

# C=100: Strict, complex boundary
svm = SVC(kernel='linear', C=100)
```

**Q6: How does a Decision Tree decide where to split?**

**A6:**

**Splitting Criteria:**

**1. Gini Impurity:**
\[
G = 1 - \sum_{i=1}^{C} p_i^2
\]

Example:
```
Node with 30 Setosa, 20 Versicolor:
G = 1 - (30/50)² - (20/50)² = 1 - 0.36 - 0.16 = 0.48

Pure node (all Setosa):
G = 1 - (50/50)² = 0
```

**2. Information Gain (Entropy-based):**
\[
H = -\sum_{i=1}^{C} p_i \log_2(p_i)
\]
\[
IG = H_{\text{parent}} - \sum \frac{N_{\text{child}}}{N_{\text{parent}}} H_{\text{child}}
\]

**Splitting Process:**
```python
# For each feature:
#   For each possible split value:
#     Compute information gain
# Choose split with highest gain
```

**Example (Iris):**
```
Split on petal_length ≤ 2.45:
  Left: 50 Setosa (pure) → G=0
  Right: 50 Versicolor, 50 Virginica → G=0.5
  
Information Gain: High! (Setosa perfectly separated)
```

**Stopping Criteria:**
- Max depth reached
- Min samples per leaf
- No more information gain
- All samples same class

---

### Implementation Questions

**Q7: Why set random_state in train_test_split?**

**A7:**

**Purpose:** Reproducibility

**Without random_state:**
```python
# Run 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Test accuracy: 96%

# Run 2 (same code)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Test accuracy: 98% (different split!)
```

**With random_state:**
```python
# Run 1
X_train, X_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Test accuracy: 97%

# Run 2 (same code)
X_train, X_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Test accuracy: 97% (same split!)
```

**Benefits:**
1. **Debugging:** Same results across runs
2. **Comparison:** Fair model comparison
3. **Collaboration:** Others can reproduce results
4. **Reporting:** Consistent performance metrics

**Note:** 42 is arbitrary (popular convention), any integer works

**Q8: How would you visualize SVM decision boundaries?**

**A8:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Use only 2 features for visualization
X_2d = X[:, [2, 3]]  # petal_length, petal_width
y = iris.target

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='auto')
svm.fit(X_2d, y)

# Create mesh grid
h = 0.02  # Step size
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict for each point in mesh
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', edgecolors='black')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('SVM Decision Boundary (RBF Kernel)')
plt.show()
```

**Observations:**
- **Linear Kernel:** Straight decision boundaries
- **Polynomial Kernel:** Curved boundaries
- **RBF Kernel:** Smooth, flexible boundaries
- **Support Vectors:** Points on or near boundary

**Q9: Implement cross-validation manually.**

**A9:**

```python
from sklearn.model_selection import KFold
import numpy as np

def manual_cross_validation(X, y, model, k=5):
    """
    Perform k-fold cross-validation manually.
    
    Args:
        X: Features
        y: Labels
        model: Sklearn model
        k: Number of folds
    
    Returns:
        List of accuracy scores
    """
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
        
        print(f"Fold {fold+1}: {score:.4f}")
    
    print(f"\nMean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores

# Usage
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
scores = manual_cross_validation(X, y, svm, k=5)
```

---

### Project-Specific Questions

**Q10: Which model performed best and why?**

**A10:**

**Best Models (typically):**
1. **SVM with RBF kernel:** 98-100%
2. **Logistic Regression:** 96-98%
3. **Naive Bayes:** 95-97%

**Why SVM Excels:**
1. **Non-linear Boundary:** RBF kernel captures non-linear relationship between Versicolor and Virginica
2. **Margin Maximization:** Robust to outliers
3. **Small Dataset:** SVM works well with limited data
4. **High-Dimensional:** Effective even when features > samples

**Why Iris is "Easy":**
1. **Well-Separated Classes:** Setosa distinctly different
2. **Low Dimensionality:** Only 4 features
3. **No Missing Values:** Clean data
4. **Balanced Classes:** 50 samples each

**Q11: How would you improve this project?**

**A11:**

**1. Hyperparameter Optimization:**
```python
from sklearn.model_selection import GridSearchCV

# SVM hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
```

**2. Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

# More robust than single train-test split
scores = cross_val_score(model, X, y, cv=10)
print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**3. Feature Engineering:**
```python
# Create interaction features
df['petal_area'] = df['petal_length'] * df['petal_width']
df['sepal_area'] = df['sepal_length'] * df['sepal_width']
df['petal_sepal_ratio'] = df['petal_length'] / df['sepal_length']
```

**4. Ensemble Methods:**
```python
from sklearn.ensemble import VotingClassifier

# Combine multiple models
voting_clf = VotingClassifier(
    estimators=[
        ('svm', SVC(kernel='rbf', probability=True)),
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression())
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
```

**5. Dimensionality Reduction:**
```python
from sklearn.decomposition import PCA

# Visualize in 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

---

## Additional Resources

**Original Paper:**
- Fisher, R.A. (1936): "The Use of Multiple Measurements in Taxonomic Problems"

**Dataset:**
- UCI Machine Learning Repository: Iris Dataset
- Scikit-learn: `sklearn.datasets.load_iris()`

**Tutorials:**
- Scikit-learn Documentation: Classification Examples
- Kaggle: Iris Flower Classification

**Books:**
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
