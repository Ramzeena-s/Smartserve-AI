
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
file_path = "messy-food-waste-prediction-dataset/train.csv"
data = pd.read_csv(file_path)

# 2. Prepare the target and input features
data["date"] = pd.to_datetime(data["date"], errors="coerce")
data["year"] = data["date"].dt.year
data["month"] = data["date"].dt.month
data["day_number"] = data["date"].dt.day

# Remove rows without a target value
data = data.dropna(subset=["food_waste_kg"])

# Features we will use for prediction
features = [
    "meals_served",
    "kitchen_staff",
    "temperature_C",
    "humidity_percent",
    "day_of_week",
    "special_event",
    "past_waste_kg",
    "staff_experience",
    "waste_category",
    "year",
    "month",
    "day_number"
]

X = data[features]
y = data["food_waste_kg"]

# 3. Identify numerical and categorical columns
numeric_features = [
    "meals_served",
    "kitchen_staff",
    "temperature_C",
    "humidity_percent",
    "past_waste_kg",
    "year",
    "month",
    "day_number"
]

categorical_features = [
    "day_of_week",
    "special_event",
    "staff_experience",
    "waste_category"
]

# 4. Fill missing values and encode text columns
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# 6. Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
print("Training SmartServe AI model...")
model.fit(X_train, y_train)

# 8. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
print(f"R-squared (R2): {r2:.3f}")

# 9. Save the trained model
joblib.dump(model, "food_waste_model.pkl")

print("\nModel saved successfully as food_waste_model.pkl")
print("You can now use this model in your website.")
