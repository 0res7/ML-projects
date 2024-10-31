import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# load data (added files in the same directory)::
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#Feature Engineering (Calculate BMI Before Preprocessing)
train_data['bmi'] = train_data['weight_in_lbs'] / ((train_data['height_in_cm'] / 100) ** 2)
test_data['bmi'] = test_data['weight_in_lbs'] / ((test_data['height_in_cm'] / 100) ** 2)


# basic info::
# print("Train data shape:", train_data.shape)
# print("Test data shape:", test_data.shape)
# print(train_data.head().to_markdown(index=False, numalign="left", stralign="left"))

# check data types::
# print("\nData types:")
# print(train_data.dtypes)

# check for missing values::
# print("\nMissing values in train data:")
# print(train_data.isnull().sum())

# print("\nMissing values in test data:")
# print(test_data.isnull().sum())

# summary statistics::
# print("\nSummary statistics of train data:")
# print(train_data.describe().to_markdown())

# check distribution of target variable::
# print("\nDistribution of triglyceride_lvl:")
# print(train_data['triglyceride_lvl'].describe())

# check unique values for categorical columns::
if False:  # Set to True if you want to see the unique values
    for col in train_data.select_dtypes(include='object'):
        print(f'\nUnique values for {col}:')

# 1. Outlier Handling:

# function to cap outliers at the 99th percentile::
def cap_outliers(df, columns):
    for col in columns:
        percentile_99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=percentile_99)
    return df

# define columns to check for outliers::
outlier_cols = ['triglyceride_lvl', 'total_cholestrol', 'good_cholestrol_lvl', 'bad_cholestrol_lvl', 'liver_enzyme_lvl1']

# cap outliers in the training data::
train_data = cap_outliers(train_data, outlier_cols)


# 2. Splitting Features and Target
X = train_data.drop(columns=['triglyceride_lvl', 'candidate_id'])
y = train_data['triglyceride_lvl']

# 3. Splitting the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Encoding and Feature Scaling:

# Preprocessor for Encoding and Scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['gender', 'smoking_habit', 'drinking_habit', 'residential_area']),
        ('ordinal_left', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]),
         ['can_hear_left_ear']),
        ('ordinal_right', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]),
         ['can_hear_right_ear'])
    ],
    remainder='passthrough'  # Passthrough for numerical columns
)

# Pipeline for Imputation and Scaling after Encoding
imputer_scaler = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combined Preprocessing Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer_scaler', imputer_scaler)
])

# Fit and transform the training data
X_train_processed = full_pipeline.fit_transform(X_train)
X_val_processed = full_pipeline.transform(X_val)

# Get feature names
feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Convert processed data back to DataFrames
X_train = pd.DataFrame(X_train_processed, columns=feature_names)
X_val = pd.DataFrame(X_val_processed, columns=feature_names)

# Apply the same transformations to the test data
X_test = test_data.drop(columns=['candidate_id'])
X_test_processed = full_pipeline.transform(X_test)
X_test = pd.DataFrame(X_test_processed, columns=feature_names)

# Models to test
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Hyperparameter grids for tuning
param_distributions = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
}

# Train and evaluate models
best_model = None
best_score = -np.inf

for name, model in models.items():
    # Train and evaluate each model
    if name in param_distributions:
        # Perform randomized search for hyperparameter tuning if applicable
        random_search = RandomizedSearchCV(model, param_distributions=param_distributions[name], n_iter=25,
                                           scoring='neg_mean_absolute_error', cv=4, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        best_score = -random_search.best_score_

    else:
        # Train the model without hyperparameter tuning if not applicable
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred_val)
        score = max(0, 100 - mae)

        if score > best_score:
            best_model = model
            best_score = score

    # Print evaluation results
    print(f'{name} MAE on Validation Set: {mae}')
    print(f'{name} Score on Validation Set: {score}')

# Prediction on the Test Set with the best model
y_pred_test = best_model.predict(X_test)

# Create submission file
# submission = pd.DataFrame({"candidate_id": test_data["candidate_id"], "triglyceride_lvl": y_pred_test})
# submission.to_csv("submission.csv", index=False)