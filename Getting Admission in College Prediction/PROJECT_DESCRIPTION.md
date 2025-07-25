
# Project Description: Admission Prediction

## Objective
The main objective of this project is to develop a machine learning model that can predict a student's chance of admission to a graduate program based on their academic profile. This is a regression problem, where the goal is to predict a continuous value representing the probability of admission.

## Dataset Used
The project uses a dataset that contains several features related to a student's academic performance and other factors that are considered for admission to a graduate program. The dataset includes the following features:

- **GRE Score**: Graduate Record Examination score
- **TOEFL Score**: Test of English as a Foreign Language score
- **University Rating**: Rating of the university the student is applying to
- **SOP**: Statement of Purpose strength
- **LOR**: Letter of Recommendation strength
- **CGPA**: Cumulative Grade Point Average
- **Research**: Whether the student has research experience (0 or 1)
- **Chance of Admit**: The probability of admission (target variable)

## ML Pipeline
The machine learning pipeline in this project can be summarized as follows:

1.  **Data Loading and Exploration**: The data is loaded from a CSV file into a pandas DataFrame. Exploratory Data Analysis (EDA) is performed to understand the data distribution, check for missing values, and visualize the relationships between different features.

2.  **Data Cleaning**: The notebook removes the 'Serial No.' column, as it is not relevant for the prediction task.

3.  **Model Training and Evaluation**:
    - The notebook uses `GridSearchCV` to find the best model and hyperparameters for this regression problem. The models considered are:
        - Linear Regression
        - Lasso Regression
        - Support Vector Regressor (SVR)
        - Decision Tree Regressor
        - Random Forest Regressor
        - K-Nearest Neighbors (KNN) Regressor
    - Linear Regression is identified as the best performing model.
    - The final model is trained on the training data and evaluated on the test data using the R-squared score.

## Technical Stack
- **Programming Language**: Python
- **Libraries**:
    - **Pandas**: For data manipulation and analysis.
    - **NumPy**: For numerical operations.
    - **Matplotlib**: For data visualization.
    - **Scikit-learn**: For implementing machine learning algorithms, preprocessing, and evaluation.

## Key Insights
- The dataset is clean and does not contain any missing values.
- The features are all numerical, which makes it easy to use them in a machine learning model.
- The Linear Regression model achieves a high R-squared score of 0.82, indicating that it is a good model for this regression task.
- The project also includes a function for making predictions on new data, which demonstrates how the model can be used in a real-world scenario.
