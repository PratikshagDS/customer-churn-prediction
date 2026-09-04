import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load processed fraud data
data = pd.read_csv(
    "../data/processed/fraud_data.csv"
)

# Features used by the fraud model
features = [
    "Purchase_Amount",
    "Transaction_Count"
]

X = data[features]
y = data["Fraud"]


# Handle missing values
X = X.copy()

X["Purchase_Amount"] = X["Purchase_Amount"].fillna(
    X["Purchase_Amount"].median()
)

X["Transaction_Count"] = X["Transaction_Count"].fillna(
    X["Transaction_Count"].median()
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
fraud_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
fraud_model.fit(
    X_train,
    y_train
)


# Save model
joblib.dump(
    fraud_model,
    "../models/fraud_model.pkl"
)

print("Fraud model saved successfully.")