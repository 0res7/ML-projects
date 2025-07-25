# Interview Prep: Credit Card Fraud Detection

This document covers key aspects of the Credit Card Fraud Detection project, designed to help you prepare for technical interviews.

## Q1: What ML algorithm is used, and why?

**Answer:**

The project uses **Logistic Regression**. Here’s why it’s a suitable choice:

*   **Interpretability:** Logistic Regression is a linear model that is easy to interpret. The coefficients of the features indicate their importance in predicting the outcome, which is valuable for understanding why a transaction is flagged as fraudulent.
*   **Efficiency:** It is computationally inexpensive to train and predict, making it a good baseline model for large datasets.
*   **Effectiveness for Binary Classification:** The problem of fraud detection is a binary classification task (fraud vs. non-fraud), and Logistic Regression is well-suited for this.

See the implementation in [`model.py`](./model.py):
```python
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)
```

## Q2: How is data preprocessed and split?

**Answer:**

1.  **Feature Scaling:** The `Amount` feature is normalized using `StandardScaler` to have a mean of 0 and a standard deviation of 1. This is important for distance-based algorithms and to prevent features with large scales from dominating the model.
2.  **Feature Dropping:** The `Time` and original `Amount` features are dropped. `Time` may not be a stable predictor, and `Amount` is replaced by its normalized version.
3.  **Data Splitting:** The data is split into training (70%) and testing (30%) sets using `train_test_split`. `stratify=y` is used to ensure that the proportion of fraudulent and non-fraudulent transactions is the same in both the training and testing sets, which is crucial for imbalanced datasets.

See the implementation in [`model.py`](./model.py):
```python
df['normalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Time', 'Amount'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

## Q3: What evaluation metrics are used, and why?

**Answer:**

The model is evaluated using a **Classification Report** and a **Confusion Matrix**.

*   **Classification Report:** This includes:
    *   **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. High precision is important to minimize false positives (flagging legitimate transactions as fraudulent).
    *   **Recall (Sensitivity):** The ratio of correctly predicted positive observations to all observations in the actual class. High recall is crucial to minimize false negatives (failing to detect fraudulent transactions).
    *   **F1-Score:** The weighted average of Precision and Recall. It is a good metric for imbalanced datasets.
*   **Confusion Matrix:** This provides a detailed breakdown of correct and incorrect classifications for each class, which is useful for understanding the types of errors the model is making.

These metrics are more informative than accuracy for imbalanced datasets like fraud detection.

## Q4: Possible improvements or alternatives?

**Answer:**

*   **Alternative Models:**
    *   **Random Forest or Gradient Boosting (XGBoost, LightGBM):** These ensemble models often perform better on tabular data and can capture more complex relationships.
    *   **Neural Networks:** A simple neural network could also be effective, especially with a large amount of data.
*   **Handling Imbalanced Data:**
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** This technique can be used to oversample the minority class (fraudulent transactions) to create a more balanced dataset.
    *   **Class Weights:** Adjusting the `class_weight` parameter in the model to penalize misclassifications of the minority class more heavily.
*   **Feature Engineering:**
    *   Creating new features from the `Time` feature, such as time of day or day of the week, could improve model performance.

## Q5: Real-world applications?

**Answer:**

This model can be applied to:

*   **Real-time Fraud Detection:** Flagging suspicious transactions as they occur to prevent financial loss.
*   **E-commerce:** Preventing fraudulent purchases on online platforms.
*   **Banking:** Identifying and preventing fraudulent loan applications or credit card applications.

## Q6: Technical challenges?

**Answer:**

*   **Imbalanced Data:** Fraudulent transactions are typically a very small percentage of all transactions, which can make it difficult to train a model that performs well.
*   **Scalability:** The model needs to be able to handle a large volume of transactions in real-time.
*   **Evolving Fraud Patterns:** Fraudsters constantly change their tactics, so the model needs to be continuously monitored and updated.

## Conceptual Questions

### Q7: What is the difference between precision and recall?

**Answer:**

*   **Precision** measures the accuracy of positive predictions. It answers the question: "Of all the transactions we flagged as fraudulent, how many were actually fraudulent?"
*   **Recall** measures the ability of the model to find all the positive samples. It answers the question: "Of all the actual fraudulent transactions, how many did we correctly identify?"

In fraud detection, there is often a trade-off between precision and recall. A model with high recall will catch more fraudulent transactions but may also have more false positives. A model with high precision will have fewer false positives but may miss more fraudulent transactions.

### Q8: How does regularization work in Logistic Regression?

**Answer:**

Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. In Logistic Regression, the two most common types of regularization are:

*   **L1 Regularization (Lasso):** Adds a penalty equal to the absolute value of the magnitude of the coefficients. This can result in some coefficients being set to zero, which can be used for feature selection.
*   **L2 Regularization (Ridge):** Adds a penalty equal to the square of the magnitude of the coefficients. This shrinks the coefficients towards zero but does not set them to zero.

The `LogisticRegression` model in scikit-learn uses L2 regularization by default.

### Q9: What is the significance of the `random_state` parameter?

**Answer:**

The `random_state` parameter is used to ensure the reproducibility of results. In many machine learning algorithms, there are random processes involved (e.g., splitting data, initializing model parameters). By setting `random_state` to a specific integer, we ensure that the same sequence of random numbers is generated each time the code is run, leading to the same results. This is important for debugging, comparing models, and ensuring that results can be replicated.
