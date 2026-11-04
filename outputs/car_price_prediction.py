
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------
# 1. Create output folder
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# 2. Load dataset
# ---------------------------------

DATA_PATH = os.path.join(BASE_DIR, "car data.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# ---------------------------------
# 3. Data cleaning
# ---------------------------------

df.columns = df.columns.str.strip()

df = df.drop_duplicates()
df = df.dropna(subset=["Selling_Price"])

# Remove unnecessary spaces in text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

cleaned_path = os.path.join(
    OUTPUT_DIR, "cleaned_car_data.csv"
)

df.to_csv(cleaned_path, index=False)

# ---------------------------------
# 4. Separate features and target
# ---------------------------------

X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# ---------------------------------
# 5. Data preprocessing
# ---------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ---------------------------------
# 6. Build Random Forest model
# ---------------------------------

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# ---------------------------------
# 7. Split dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining the model...")

# ---------------------------------
# 8. Train the model
# ---------------------------------

model.fit(X_train, y_train)

print("Model training completed!")

# ---------------------------------
# 9. Make predictions
# ---------------------------------

y_pred = model.predict(X_test)

# ---------------------------------
# 10. Model evaluation
# ---------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

evaluation = f"""
CAR PRICE PREDICTION - MODEL EVALUATION

Model: Random Forest Regressor

Total Records: {len(df)}
Training Records: {len(X_train)}
Testing Records: {len(X_test)}

Mean Absolute Error (MAE): {mae:.4f}

Mean Squared Error (MSE): {mse:.4f}

Root Mean Squared Error (RMSE): {rmse:.4f}

R-squared Score (R2): {r2:.4f}

Model training and evaluation completed successfully.
"""

print(evaluation)

with open(
    os.path.join(OUTPUT_DIR, "model_evaluation.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(evaluation)

# ---------------------------------
# 11. Save predictions
# ---------------------------------

predictions = pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": y_pred
})

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "predictions.csv"),
    index=False
)

# ---------------------------------
# 12. Actual vs Predicted graph
# ---------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=300
)

plt.close()

# ---------------------------------
# 13. Feature importance
# ---------------------------------

trained_preprocessor = model.named_steps["preprocessor"]
trained_forest = model.named_steps["regressor"]

feature_names = trained_preprocessor.get_feature_names_out()

importance = trained_forest.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False
)

# Feature importance graph
top_features = feature_importance.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_importance.png"),
    dpi=300
)

plt.close()

# ---------------------------------
# 14. Final result
# ---------------------------------

print("\nAll tasks completed successfully!")

print("\nOutput files saved in:")
print(OUTPUT_DIR)

print("\nGenerated files:")
for filename in os.listdir(OUTPUT_DIR):
    print(filename)
