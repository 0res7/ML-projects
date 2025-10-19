# Interview Preparation: Mechanisms of Action (MoA) Prediction

## 1. Project Overview

**Problem Statement:** Predict the biological mechanism of action (MoA) of drugs based on gene expression and cell viability data from cellular experiments.

**Objective:** Build multi-label classification models to identify which biological pathways a drug affects, accelerating drug discovery and understanding drug behavior.

**Challenge:** 
- **Multi-label:** Each drug can have multiple MoAs (206 targets)
- **High-dimensional:** 772 gene expression features + 100 cell viability features
- **Imbalanced:** Some MoAs rare, others common

**Real-World Impact:** Helps pharmaceutical companies understand drug mechanisms, predict side effects, and repurpose existing drugs.

---

## 2. Technical Concepts

### Multi-Label Classification
- **Difference from Multi-Class:** Each sample can have 0, 1, or multiple labels
- **Example:** Drug affects both "protease inhibitor" AND "kinase inhibitor"
- **Output:** Binary vector of length 206 (one per MoA)

### Feature Types
- **Gene Expression (g- features):** 772 continuous values measuring gene activity
- **Cell Viability (c- features):** 100 continuous values measuring cell health

### Algorithms
- **Neural Networks:** Handle high dimensionality
- **XGBoost/LightGBM:** With multi-label wrapper
- **One-vs-Rest:** Separate classifier per MoA
- **Classifier Chains:** Model label dependencies

---

## 3. Mathematical Foundations

### Multi-Label Loss
**Binary Cross-Entropy per Label:**
\[
L = -\frac{1}{K}\sum_{k=1}^{K}\sum_{i=1}^{N}[y_{ik}\log(\hat{y}_{ik}) + (1-y_{ik})\log(1-\hat{y}_{ik})]
\]
where \(K=206\) MoAs, \(N\) samples.

### Sigmoid Activation (Independent Labels)
\[
\hat{y}_k = \sigma(z_k) = \frac{1}{1 + e^{-z_k}}
\]

Each output independent (unlike softmax which sums to 1).

### Evaluation Metrics

**Log Loss (Multi-Label):**
\[
\text{LogLoss} = -\frac{1}{N \times M}\sum_{i,j} [y_{ij}\log(p_{ij}) + (1-y_{ij})\log(1-p_{ij})]
\]

**Hamming Loss:**
\[
\text{Hamming} = \frac{1}{N \times K}\sum_{i,j} \mathbb{1}(y_{ij} \neq \hat{y}_{ij})
\]

**F1-Score (Micro/Macro):**
- **Micro:** Aggregate all predictions, compute F1
- **Macro:** Average F1 across all labels

---

## 4. Implementation Details

### Data Structure
```python
# train_features.csv
- sig_id: Sample identifier
- cp_type: Compound or control perturbation
- cp_time: Treatment duration (24h, 48h, 72h)
- cp_dose: Dose level (D1, D2)
- g-0 to g-771: Gene expression features
- c-0 to c-99: Cell viability features

# train_targets_scored.csv
- sig_id: Sample identifier
- 206 binary columns (one per MoA)
```

### Workflow
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

# Load data
features = pd.read_csv('train_features.csv')
targets = pd.read_csv('train_targets_scored.csv')

# Merge
train = features.merge(targets, on='sig_id')

# Remove control samples (no MoA)
train = train[train['cp_type'] != 'ctl_vehicle']

# Encode categorical
train['cp_time'] = train['cp_time'].map({24: 0, 48: 1, 72: 2})
train['cp_dose'] = train['cp_dose'].map({'D1': 0, 'D2': 1})

# Separate features and targets
feature_cols = [col for col in train.columns if col.startswith('g-') or col.startswith('c-')]
target_cols = [col for col in train.columns if col not in ['sig_id', 'cp_type'] + feature_cols]

X = train[feature_cols + ['cp_time', 'cp_dose']]
y = train[target_cols]

# Preprocessing
# Quantile transformation (robust to outliers)
qt = QuantileTransformer(n_quantiles=100, random_state=42, output_distribution='normal')
X_transformed = qt.fit_transform(X)

# Optional: PCA for dimensionality reduction
pca = PCA(n_components=500)
X_pca = pca.fit_transform(X_transformed)
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42
)

# Build Neural Network
model = Sequential([
    Dense(1024, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(206, activation='sigmoid')  # 206 independent outputs
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

# Predict
y_pred = model.predict(X_val)

# Evaluate
from sklearn.metrics import log_loss
score = log_loss(y_val.values.flatten(), y_pred.flatten())
print(f"Log Loss: {score:.4f}")
```

### Handling Class Imbalance
```python
# Some MoAs very rare (<1% positive)
# Solutions:

# 1. Class weights
from sklearn.utils.class_weight import compute_sample_weight
weights = compute_sample_weight('balanced', y_train)

# 2. Focal Loss (focuses on hard examples)
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal = alpha * tf.pow(1 - p_t, gamma) * bce
    return tf.reduce_mean(focal)

# 3. SMOTE (per label)
from imblearn.over_sampling import SMOTE
```

---

## 5. Outcomes & Results

### Typical Performance
- **Log Loss:** 0.015-0.020 (Kaggle competition)
- **Baseline:** ~0.03 (predicting all zeros)

### Key Insights
- **Gene Expression Most Important:** More predictive than cell viability
- **Neural Networks Best:** Handle high dimensions well
- **Ensemble Helps:** Combine multiple models

---

## 6. Interview Questions & Answers

**Q1: What is multi-label classification vs multi-class?**

**A1:**

**Multi-Class:**
- One label per sample
- Example: Iris (Setosa OR Versicolor OR Virginica)
- Output: Softmax (probabilities sum to 1)

**Multi-Label:**
- Multiple labels per sample
- Example: MoA (kinase AND protease inhibitor)
- Output: Sigmoid (independent probabilities)

**Q2: Why use sigmoid instead of softmax in output layer?**

**A2:**

**Sigmoid:**
```python
output = Dense(206, activation='sigmoid')
# Each output independent: [0.8, 0.3, 0.6, ...]
# Can all be high or all be low
```

**Softmax (Wrong for Multi-Label):**
```python
output = Dense(206, activation='softmax')
# Outputs sum to 1: [0.4, 0.2, 0.15, ...]
# Forces competition between labels (wrong!)
```

**Q3: How does PCA help with this high-dimensional data?**

**A3:** **Dimensionality Reduction**

**Original:** 872 features (772 genes + 100 cell viability)
**After PCA:** 500 components (retain 95% variance)

**Benefits:**
1. Faster training (fewer features)
2. Reduces multicollinearity
3. Removes noise
4. Visualization (first 2-3 components)

**Trade-off:** Lose interpretability (components are linear combinations)

**Q4: What is QuantileTransformer and why use it?**

**A4:**

**Purpose:** Transform features to follow normal distribution

**Robust to Outliers:**
- Unlike StandardScaler, not affected by extreme values
- Maps to uniform then to Gaussian distribution

```python
qt = QuantileTransformer(output_distribution='normal')
X_transformed = qt.fit_transform(X)
```

**When to Use:**
- Skewed distributions
- Outliers present
- Non-linear relationships

**Q5: How would you deploy this model for drug screening?**

**A5:**

**Production Pipeline:**
```python
# 1. High-throughput screening
# Test compound on cells → Gene expression data

# 2. Preprocess
X_new = qt.transform(new_compound_data)

# 3. Predict MoAs
moa_probs = model.predict(X_new)

# 4. Threshold
threshold = 0.5
predicted_moas = (moa_probs > threshold).astype(int)

# 5. Report
moa_names = [moa for i, moa in enumerate(moa_list) if predicted_moas[0][i] == 1]
print(f"Predicted MoAs: {moa_names}")
print(f"Probabilities: {moa_probs[0][predicted_moas[0] == 1]}")
```

**Applications:**
- Drug repurposing
- Side effect prediction
- Mechanism understanding

---

## Additional Resources

**Kaggle Competition:** "Mechanisms of Action (MoA) Prediction"
**Papers:** Multi-label classification techniques
**Biology:** Drug mechanism of action databases

