import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error

# Load the datasets
train_df = pd.read_csv('/Users/saikrishna_gajula/ML/ML-projects/water_potability/dataset/train.csv')
test_df = pd.read_csv('/Users/saikrishna_gajula/ML/ML-projects/water_potability/dataset/test.csv')

# Define the columns to process
columns_to_process = ["ph", "Hardness", "Solids", "Turbidity"]

# Loop through the columns and extract numeric values
for col in columns_to_process:
    train_df[col] = train_df[col].astype(str).str.extract(r"(\d+\.\d+)").astype(float)
    test_df[col] = test_df[col].astype(str).str.extract(r"(\d+\.\d+)").astype(float)

# Split the train dataset into features and target variables
X = train_df.drop(["Potability", "Index"], axis=1)
y = train_df["Potability"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the target variable for the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model's performance on the validation set
median_ae = median_absolute_error(y_val, y_pred_val)
score = max(0, 100 * (1 - median_ae))

print(f"Median Absolute Error: {median_ae}")
print(f"Score: {score}")

# Predict the target variable for the entire training dataset
y_pred_train = model.predict(X)

# Calculate the Median Absolute Error between the predicted and true values
median_ae = median_absolute_error(y, y_pred_train)

# Calculate the score using the formula: `score = max(0, 100 * (1 - median_absolute_error))`
score = max(0, 100 * (1 - median_ae))

# Print the performance metrics
print(f'Median Absolute Error: {median_ae}')
print(f'Score: {score}')

# Display the first 5 rows of the `train_df` DataFrame
print("Train DataFrame Head:")
print(train_df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Print the column names and their data types for the `train_df` DataFrame
print("\nTrain DataFrame Info:")
train_df.info()