import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def run_model(data_path):
    # Load data
    df = pd.read_csv(data_path)

    # Preprocessing
    df['normalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
    df = df.drop(['Time', 'Amount'], axis=1)

    # Split data
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Train model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    run_model('creditcard.csv')
