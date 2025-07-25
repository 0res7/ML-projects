
# Project Description: Diabetes Prediction

## Objective
The main objective of this project is to develop a machine learning model that can predict whether a person has diabetes or not based on several diagnostic measurements. This is a binary classification problem, where the outcome is either 0 (no diabetes) or 1 (has diabetes).

## Dataset Used
The project utilizes a dataset from Kaggle, which contains several medical predictor variables and one target variable, `Outcome`. The dataset includes the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1)

## ML Pipeline
The machine learning pipeline in this project can be summarized as follows:

1.  **Data Loading and Exploration**: The data is loaded from a CSV file into a pandas DataFrame. Exploratory Data Analysis (EDA) is performed to understand the data distribution, check for missing values, and visualize the relationships between different features.

2.  **Data Cleaning**: The notebook identifies that some of the features, such as 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI', have zero values, which are likely missing values. These zero values are replaced with NaN and then imputed using the mean or median of the respective columns.

3.  **Feature Scaling**: The features are scaled using `StandardScaler` to ensure that all features have a mean of 0 and a standard deviation of 1. This is an important step for many machine learning algorithms, as it prevents features with larger scales from dominating the learning process.

4.  **Model Training and Evaluation**:
    - The notebook uses `GridSearchCV` to find the best model and hyperparameters for this classification problem. The models considered are:
        - Logistic Regression
        - Decision Tree Classifier
        - Random Forest Classifier
        - Support Vector Machine (SVM)
    - The Random Forest Classifier is identified as the best performing model.
    - The final model is trained on the training data and evaluated on the test data using a confusion matrix, accuracy score, and classification report.

## Technical Stack
- **Programming Language**: Python
- **Libraries**:
    - **Pandas**: For data manipulation and analysis.
    - **NumPy**: For numerical operations.
    - **Matplotlib & Seaborn**: For data visualization.
    - **Scikit-learn**: For implementing machine learning algorithms, preprocessing, and evaluation.

## Key Insights
- The dataset contains some missing values, which are represented as zeros. These need to be handled appropriately to avoid biased results.
- The Random Forest Classifier achieves a high accuracy of 98.75% on the test set, indicating that it is a suitable model for this classification task.
- The confusion matrix shows that the model is able to correctly classify most of the instances, with only a few false negatives.
- The project also includes a function for making predictions on new data, which demonstrates how the model can be used in a real-world scenario.
