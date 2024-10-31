import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load data (added files in the same directory)::
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

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
for col in train_data.select_dtypes(include='object'):
     print(f'\nUnique values for {col}:')
    # print(train_data[col].unique())

# 1. Outlier Handling:
# define columns to check for outliers::
outlier_cols = ['triglyceride_lvl', 'total_cholestrol', 'good_cholestrol_lvl', 'bad_cholestrol_lvl', 'liver_enzyme_lvl1']

# function to cap outliers at the 99th percentile::
def cap_outliers(df, columns):
    for col in columns:
        percentile_99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=percentile_99)
    return df

# cap outliers in the training data::
train_data = cap_outliers(train_data, outlier_cols)


# 2. Encoding Categorical Features:
X_train = train_data.drop(columns=['triglyceride_lvl', 'candidate_id'])
y_train = train_data['triglyceride_lvl']

# create column transformer to handle different encoding types::
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), ['gender', 'smoking_habit', 'drinking_habit', 'residential_area']),
        ('ordinal_left', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]), ['can_hear_left_ear']),
        ('ordinal_right', OrdinalEncoder(categories=[['Normal', 'Slightly Defective', 'Highly Defective']]), ['can_hear_right_ear'])
    ],
    remainder='passthrough'  # Passthrough for numerical columns
)

# apply the column transformer to the data::
X_train_encoded = preprocessor.fit_transform(X_train)

# convert the encoded data back to a DataFrame for easier manipulation later::
X_train = pd.DataFrame(X_train_encoded, columns=preprocessor.get_feature_names_out())

# create test data matrix (excluding candidate_id and triglyceride_lvl)::
X_test = test_data.drop(columns=['candidate_id'])

# apply the same transformations to the test data (use 'transform', not 'fit_transform')::
X_test_encoded = preprocessor.transform(X_test)
X_test = pd.DataFrame(X_test_encoded, columns=preprocessor.get_feature_names_out())

# 3. Impute Missing Values in numerical columns
# separate numerical columns::
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# imputer for numerical columns::
imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])

# 4. Feature Scaling:
# create scaler object::
scaler = StandardScaler()

# fit the scaler to the training data and transform::
train_data_scaled = scaler.fit_transform(X_train)

# transform the test data::
test_data_scaled = scaler.transform(X_test)

# convert back to DataFrames::
X_train = pd.DataFrame(train_data_scaled, columns=X_train.columns)
X_test = pd.DataFrame(test_data_scaled, columns=X_test.columns)

# splitting the data into train and validation sets::
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model Building::
model_lr = LinearRegression()  # Create a Linear Regression model
model_lr.fit(X_train_final, y_train_final)  # Train the model on the training data

# Prediction and Evaluation on the Validation Set::
y_pred_val = model_lr.predict(X_val)
mae = mean_absolute_error(y_val, y_pred_val)
score = max(0, 100 - mae)

print("Linear Regression MAE on Validation Set:", mae)
print("Linear Regression Score on Validation Set:", score)

# Prediction on the Test Set ::
y_pred_test = model_lr.predict(X_test)

# Create submission_file::
submission = pd.DataFrame({"candidate_id": test_data["candidate_id"], "triglyceride_lvl": y_pred_test})
submission.to_csv("submission.csv", index=False)

# output:
#
# /Users/saikrishna_gajula/PycharmProjects/triglyceride_prediction/.venv/bin/python /Users/saikrishna_gajula/PycharmProjects/triglyceride_prediction/main.py
#
# Unique values for candidate_id:
#
# Unique values for gender:
#
# Unique values for can_hear_left_ear:
#
# Unique values for can_hear_right_ear:
#
# Unique values for smoking_habit:
#
# Unique values for drinking_habit:
#
# Unique values for residential_area:
# Linear Regression MAE on Validation Set: 12.167073463736026
# Linear Regression Score on Validation Set: 87.83292653626397
#
# Process finished with exit code 0
