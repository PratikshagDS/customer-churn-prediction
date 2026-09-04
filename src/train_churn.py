import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load processed churn data
data = pd.read_csv(
    "../data/processed/churn_data.csv"
)

# Features used by the churn model
features = [
    "total_transactions",
    "total_spent",
    "average_purchase"
]

X = data[features]
y = data["Churn"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
churn_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
churn_model.fit(
    X_train,
    y_train
)


# Save model
joblib.dump(
    churn_model,
    "../models/churn_model.pkl"
)

print("Churn model saved successfully.")