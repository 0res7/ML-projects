import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Feature Engineering (BMI)
train_data['bmi'] = train_data['weight_in_lbs'] / ((train_data['height_in_cm'] / 100) ** 2)
test_data['bmi'] = test_data['weight_in_lbs'] / ((test_data['height_in_cm'] / 100) ** 2)

# Outlier Handling
def cap_outliers(df, columns):
    for col in columns:
        percentile_99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=percentile_99)
    return df

outlier_cols = ['triglyceride_lvl', 'total_cholestrol', 'good_cholestrol_lvl', 'bad_cholestrol_lvl', 'liver_enzyme_lvl1', 'bmi']
train_data = cap_outliers(train_data, outlier_cols)

# Splitting Data
X = train_data.drop(columns=['triglyceride_lvl', 'candidate_id'])
y = train_data['triglyceride_lvl']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_data.drop(columns=['candidate_id'])

# Preprocessing Pipeline
categorical_features = ['gender', 'smoking_habit', 'drinking_habit', 'residential_area']
ordinal_features = ['can_hear_left_ear', 'can_hear_right_ear']

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
        ('ordinal_left', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]), ['can_hear_left_ear']),
        ('ordinal_right', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]), ['can_hear_right_ear'])
    ],
    remainder='passthrough'
)

# Add Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

imputer_scaler = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Combined Preprocessing Pipeline with Polynomial Features
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', poly),
    ('imputer_scaler', imputer_scaler)
])

X_train = full_pipeline.fit_transform(X_train)
X_val = full_pipeline.transform(X_val)
X_test = full_pipeline.transform(X_test)

# Model Definitions and Hyperparameter Distributions
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

param_distributions = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    },
    'Ridge Regression': {
        'alpha': [0.1, 1.0, 10.0]
    },
    'Lasso Regression': {
        'alpha': [0.1, 1.0, 10.0]
    }
}

best_model = None
best_score = -np.inf

for name, model in models.items():
    print(f"Training {name}...")
    if name in param_distributions:
        random_search = RandomizedSearchCV(model, param_distributions=param_distributions[name], n_iter=10,
                                           scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, random_state=42)
        random_search.fit(X_train, y_train)
        best_model_instance = random_search.best_estimator_
        best_mae = -random_search.best_score_
    else:
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        best_mae = mean_absolute_error(y_val, y_pred_val)
        best_model_instance = model

    score = max(0, 100 - best_mae)

    if score > best_score:
        best_model = best_model_instance
        best_score = score

    print(f'{name} MAE on Validation Set: {best_mae}')
    print(f'{name} Score on Validation Set: {score}')

# Prediction on the Test Set with the best model
y_pred_test = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({"candidate_id": test_data["candidate_id"], "triglyceride_lvl": y_pred_test})
submission.to_csv("submission.csv", index=False)
