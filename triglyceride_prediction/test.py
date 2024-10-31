import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Feature Engineering (if needed)
# Outlier Handling (if needed)

# Split data into features and target
X_train = train_data.drop(columns=['triglyceride_lvl', 'candidate_id'])
y_train = train_data['triglyceride_lvl']
X_test = test_data.drop(columns=['candidate_id'])

# Preprocessing
# Define numerical and categorical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Define preprocessing steps for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and fit the preprocessor
preprocessed_X_train = preprocessor.fit_transform(X_train)
preprocessed_X_test = preprocessor.transform(X_test)

# Split data into train and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(preprocessed_X_train, y_train, test_size=0.2, random_state=42)

# Model Building with Hyperparameter Tuning
param_grid = {'fit_intercept': [True, False]}  # Hyperparameters to tune

lr = LinearRegression()  # Create a Linear Regression model

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_final, y_train_final)

# Best model after hyperparameter tuning
best_lr = grid_search.best_estimator_

# Prediction and Evaluation on the Validation Set
y_pred_val = best_lr.predict(X_val)
mae = mean_absolute_error(y_val, y_pred_val)
score = max(0, 100 - mae)

print("Linear Regression MAE on Validation Set:", mae)
print("Linear Regression Score on Validation Set:", score)

# Prediction on the Test Set
y_pred_test = best_lr.predict(preprocessed_X_test)

# Create submission file
submission = pd.DataFrame({"candidate_id": test_data["candidate_id"], "triglyceride_lvl": y_pred_test})
submission.to_csv("submission.csv", index=False)
