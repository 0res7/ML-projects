import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1. Load and Prepare Data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Feature Engineering (Calculate BMI Before Preprocessing)
train_data['bmi'] = train_data['weight_in_lbs'] / ((train_data['height_in_cm'] / 100) ** 2)
test_data['bmi'] = test_data['weight_in_lbs'] / ((test_data['height_in_cm'] / 100) ** 2)

# 2. Outlier Handling
outlier_cols = ['triglyceride_lvl', 'total_cholestrol', 'good_cholestrol_lvl', 'bad_cholestrol_lvl', 'liver_enzyme_lvl1', 'bmi']
train_data = cap_outliers(train_data, outlier_cols)  # Apply to train_data only


# 3. Split Features and Target (Before Any Preprocessing!)
X = train_data.drop(columns=['triglyceride_lvl', 'candidate_id'])
y = train_data['triglyceride_lvl']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Encoding and Feature Scaling:
categorical_features = ['gender', 'smoking_habit', 'drinking_habit', 'residential_area']
ordinal_features = ['can_hear_left_ear', 'can_hear_right_ear']

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
        ('ordinal_left', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]), ['can_hear_left_ear']),
        ('ordinal_right', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]), ['can_hear_right_ear'])
    ],
    remainder='passthrough'  # Passthrough numerical features
)


#Pipeline for Imputation and Scaling after Encoding
imputer_scaler = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

#Combined Preprocessing Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer_scaler', imputer_scaler)
])

# Fit and transform the training data
X_train = full_pipeline.fit_transform(X_train)
X_val = full_pipeline.transform(X_val)

# Get feature names
feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Convert back to DataFrames
X_train = pd.DataFrame(X_train, columns=feature_names)
X_val = pd.DataFrame(X_val, columns=feature_names)

# Apply the same transformations to the test data
X_test = test_data.drop(columns=['candidate_id'])
X_test_processed = full_pipeline.transform(X_test)
X_test = pd.DataFrame(X_test_processed, columns=feature_names)



# 5. Models and Hyperparameter Tuning
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Hyperparameter grids for tuning (Expanded and refined)
param_distributions = {
    'Random Forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    },
    'Ridge Regression': {
        'alpha': [0.1, 1.0, 10.0],
    },
    'Lasso Regression': {
        'alpha': [0.1, 1.0, 10.0],
    }
}


# 6. Train and Evaluate Models
best_model = None
best_score = -np.inf

for name, model in models.items():
    # Train and evaluate each model
    if name in param_distributions:
        # Perform randomized search for hyperparameter tuning if applicable
        random_search = RandomizedSearchCV(model, param_distributions=param_distributions[name], n_iter=30,
                                           scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, random_state=42)
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
submission = pd.DataFrame({"candidate_id": test_data["candidate_id"], "triglyceride_lvl": y_pred_test})
submission.to_csv("submission.csv", index=False)
