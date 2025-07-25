# Interview Prep: Customer Churn Prediction

This document covers key aspects of the Customer Churn Prediction project, designed to help you prepare for technical interviews.

## Q1: What ML algorithm is used, and why?

**Answer:**

The project uses a **Random Forest Classifier**. Here’s why it’s a suitable choice:

*   **High Accuracy:** Random Forests are ensemble models that often achieve high accuracy on a wide range of problems.
*   **Robust to Overfitting:** By averaging the predictions of multiple decision trees, Random Forests are less prone to overfitting than individual trees.
*   **Handles Non-linear Relationships:** They can capture complex, non-linear relationships between features and the target variable.
*   **Feature Importance:** Random Forests provide a measure of feature importance, which can be used to understand which factors are most influential in predicting churn.

See the implementation in [`model.py`](./model.py):
```python
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])
model.fit(X_train, y_train)
```

## Q2: How is data preprocessed and split?

**Answer:**

1.  **Preprocessing Pipeline:** A `ColumnTransformer` is used to apply different preprocessing steps to different types of columns.
    *   **Numerical Features:** `StandardScaler` is used to scale numerical features to have a mean of 0 and a standard deviation of 1.
    *   **Categorical Features:** `OneHotEncoder` is used to convert categorical features into a numerical format.
2.  **Data Splitting:** The data is split into training (80%) and testing (20%) sets using `train_test_split`. `stratify=y` is used to ensure that the proportion of churned and non-churned customers is the same in both the training and testing sets.

See the implementation in [`model.py`](./model.py):
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## Q3: What evaluation metrics are used, and why?

**Answer:**

The model is evaluated using **Accuracy** and **ROC AUC Score**.

*   **Accuracy:** The ratio of correctly predicted observations to the total observations. It is a good general-purpose metric when the classes are relatively balanced.
*   **ROC AUC Score:** The Area Under the Receiver Operating Characteristic Curve. It measures the ability of the model to distinguish between classes. An AUC of 1.0 means the model is a perfect classifier, while an AUC of 0.5 means the model is no better than random guessing. ROC AUC is particularly useful for evaluating binary classifiers on imbalanced datasets.

## Q4: Possible improvements or alternatives?

**Answer:**

*   **Alternative Models:**
    *   **Gradient Boosting (XGBoost, LightGBM):** These models often outperform Random Forests and are widely used in industry.
    *   **Logistic Regression:** A simpler, more interpretable model that could be used as a baseline.
*   **Hyperparameter Tuning:** Using `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the Random Forest model could improve performance.
*   **Feature Engineering:** Creating new features, such as the ratio of balance to salary, could provide the model with more information.

## Q5: Real-world applications?

**Answer:**

This model can be applied to:

*   **Telecommunications:** Identifying customers who are likely to switch to a competitor.
*   **Subscription Services:** Predicting which customers are likely to cancel their subscriptions.
*   **Banking:** Identifying customers who are likely to close their accounts.

## Q6: Technical challenges?

**Answer:**

*   **Data Quality:** The model's performance is highly dependent on the quality of the data. Missing values, outliers, and incorrect data can all negatively impact the model.
*   **Interpretability:** While Random Forests are powerful, they can be difficult to interpret. Understanding why the model is making certain predictions can be challenging.
*   **Feature Selection:** Identifying the most important features for predicting churn can be difficult, especially with a large number of features.

## Conceptual Questions

### Q7: What is the difference between a Random Forest and a single Decision Tree?

**Answer:**

A **Decision Tree** is a single model that makes predictions by learning a series of if/else questions about the features. It is prone to overfitting, especially if the tree is deep.

A **Random Forest** is an ensemble of many Decision Trees. It builds multiple trees on different subsets of the data and features, and then averages their predictions. This process, known as bagging, helps to reduce overfitting and improve the model's accuracy and robustness.

### Q8: What is One-Hot Encoding?

**Answer:**

One-Hot Encoding is a technique used to convert categorical variables into a numerical format that can be used by machine learning algorithms. It creates a new binary column for each category in the original feature. For each observation, the value in the column corresponding to its category is set to 1, and the values in all other columns are set to 0.

### Q9: What is the purpose of a validation set?

**Answer:**

A validation set is a subset of the training data that is not used to train the model. Instead, it is used to tune the model's hyperparameters and to get an unbiased estimate of the model's performance on unseen data. This helps to prevent overfitting and to select the best model. In this project, a test set is used for final evaluation, but a validation set would be used during the model development and tuning phase.
