import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Load processed recommendation data
data = pd.read_csv(
    "../data/processed/recommendation_data.csv"
)

# Features
X = data[
    ["Customer_ID", "Brand"]
].copy()

y = data["Interaction"]


# Encode Customer_ID
customer_encoder = LabelEncoder()
X["Customer_ID"] = customer_encoder.fit_transform(
    X["Customer_ID"]
)


# Encode Brand
brand_encoder = LabelEncoder()
X["Brand"] = brand_encoder.fit_transform(
    X["Brand"]
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
recommendation_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
recommendation_model.fit(
    X_train,
    y_train
)


# Save model
joblib.dump(
    recommendation_model,
    "../models/recommendation_model.pkl"
)

# Save encoders
joblib.dump(
    customer_encoder,
    "../models/customer_encoder.pkl"
)

joblib.dump(
    brand_encoder,
    "../models/brand_encoder.pkl"
)

print("Recommendation model saved successfully.")
print("Customer encoder saved successfully.")
print("Brand encoder saved successfully.")