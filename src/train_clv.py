import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Load processed CLV data
data = pd.read_csv(
    "../data/processed/clv_data.csv"
)

# Features used by the CLV model
features = [
    "total_transactions",
    "total_spent",
    "average_purchase"
]

X = data[features]
y = data["CLV"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
clv_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train model
clv_model.fit(
    X_train,
    y_train
)


# Save model
joblib.dump(
    clv_model,
    "../models/clv_model.pkl"
)

print("CLV model saved successfully.")