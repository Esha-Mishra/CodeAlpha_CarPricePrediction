from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Find the folder containing this Python file
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# STEP 2: Load the dataset
file_path = BASE_DIR / "car data.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# STEP 3: Clean the data
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df = df.dropna(subset=["Selling_Price"])

# Save cleaned dataset
df.to_csv(OUTPUT_DIR / "cleaned_car_data.csv", index=False)

print("\nDataset after cleaning:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# STEP 4: Select features and target
target = "Selling_Price"

X = df.drop(columns=[target])
y = df[target]

numeric_features = [
    "Year", "Present_Price", "Driven_kms", "Owner"
]

categorical_features = [
    "Car_Name", "Fuel_Type", "Selling_type", "Transmission"
]

# STEP 5: Convert text categories into numerical form
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("numerical", "passthrough", numeric_features)
    ]
)

# STEP 6: Build the regression model
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# STEP 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# STEP 8: Train the model
model.fit(X_train, y_train)
print("\nModel training completed!")

# STEP 9: Predict car prices
y_pred = model.predict(X_test)

# STEP 10: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared (R2): {r2:.3f}")

# Save actual and predicted prices
results = pd.DataFrame({
    "Actual_Selling_Price": y_test.values,
    "Predicted_Selling_Price": y_pred
})
results.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

# STEP 11: Plot actual prices versus predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
plt.close()

# STEP 12: Plot feature importance
regressor = model.named_steps["regressor"]
feature_names = model.named_steps[
    "preprocessing"
].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": regressor.feature_importances_
}).sort_values("Importance", ascending=False).head(10)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv", index=False
)

plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 10 Features Affecting Car Price Prediction")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png")
plt.close()

print("\nTop 10 important features:")
print(importance.to_string(index=False))

print("\nAll tasks completed successfully!")
print("Check the outputs folder for results and graphs.")