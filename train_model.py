
import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1. Load dataset
df = pd.read_csv("datasets/vegetable_sales.csv")

df["date"] = pd.to_datetime(df["date"])

# 2. Sort data
df = df.sort_values(
    ["vegetable", "date"]
).reset_index(drop=True)

# 3. Create historical sales features
df["previous_day_sales"] = (
    df.groupby("vegetable")["quantity_sold_kg"]
    .transform(lambda x: x.shift(1))
)

df["previous_7_day_average"] = (
    df.groupby("vegetable")["quantity_sold_kg"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["previous_30_day_average"] = (
    df.groupby("vegetable")["quantity_sold_kg"]
    .transform(
        lambda x: x.shift(1).rolling(30).mean()
    )
)

# 4. Remove rows without sufficient history
df = df.dropna().reset_index(drop=True)

print("Historical features created successfully!")

# 5. Define input features
features = [
    "vegetable",
    "price_per_kg",
    "day_of_week",
    "month",
    "is_weekend",
    "previous_day_sales",
    "previous_7_day_average",
    "previous_30_day_average"
]

target = "quantity_sold_kg"

X = df[features]
y = df[target]

# 6. Chronological train-test split
split_date = df["date"].quantile(0.8)

train_mask = df["date"] < split_date
test_mask = ~train_mask

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Encode vegetable names
preprocessor = ColumnTransformer(
    transformers=[
        (
            "vegetable_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["vegetable"]
        )
    ],
    remainder="passthrough"
)

# 8. Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            max_depth=12
        )
    )
])

# 9. Train model
model.fit(X_train, y_train)

print("\nImproved model training completed!")

# 10. Predict
predictions = model.predict(X_test)

# 11. Evaluate
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\n======= IMPROVED MODEL RESULTS =======")

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 13. Compare with a simple 7-day average baseline

baseline_predictions = X_test[
    "previous_7_day_average"
].to_numpy()

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

baseline_r2 = r2_score(
    y_test, baseline_predictions
)

print("\n======= BASELINE COMPARISON =======")

print("Random Forest MAE:", round(mae, 2))
print("7-Day Baseline MAE:", round(baseline_mae, 2))

print("Random Forest RMSE:", round(rmse, 2))
print("7-Day Baseline RMSE:", round(baseline_rmse, 2))

print("Random Forest R2:", round(r2, 4))
print("7-Day Baseline R2:", round(baseline_r2, 4))


# 12. Save improved model
joblib.dump(
    model,
    "models/demand_model.pkl"
)

print("\nImproved model saved successfully!")
