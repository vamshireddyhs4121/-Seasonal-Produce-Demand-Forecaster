
import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Project dataset settings
vegetables = {
    "Tomato": {
        "base_price": 30,
        "base_demand": 40
    },
    "Onion": {
        "base_price": 35,
        "base_demand": 35
    },
    "Potato": {
        "base_price": 25,
        "base_demand": 30
    },
    "Carrot": {
        "base_price": 40,
        "base_demand": 20
    },
    "Beans": {
        "base_price": 45,
        "base_demand": 18
    }
}

# Generate two years of daily records
dates = pd.date_range(
    start="2024-01-01",
    end="2026-11-02",
    freq="D"
)

records = []

for date in dates:

    for vegetable, details in vegetables.items():

        month = date.month
        day_of_week = date.day_of_week

        # Seasonal demand variation (illustrative)
        if month in [6, 7, 8, 9]:
            seasonal_factor = 1.15
        elif month in [12, 1, 2]:
            seasonal_factor = 1.10
        else:
            seasonal_factor = 1.0

        # Slightly higher demand on weekends
        weekend_factor = 1.15 if day_of_week >= 5 else 1.0

        # Generate selling price
        price = details["base_price"] + np.random.uniform(-5, 5)

        # Demand decreases slightly as price increases
        price_factor = 1 - (
            (price - details["base_price"])
            / details["base_price"]
        ) * 0.3

        # Generate daily quantity sold
        quantity = (
            details["base_demand"]
            * seasonal_factor
            * weekend_factor
            * price_factor
            + np.random.normal(0, 4)
        )

        quantity = max(1, round(quantity, 2))

        records.append({
            "date": date,
            "vegetable": vegetable,
            "price_per_kg": round(price, 2),
            "quantity_sold_kg": quantity
        })

# Convert records into a DataFrame
df = pd.DataFrame(records)

# Add calendar features
df["day_of_week"] = df["date"].dt.day_of_week
df["month"] = df["date"].dt.month
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Save CSV file
df.to_csv(
    "datasets/vegetable_sales.csv",
    index=False
)

print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst five records:")
print(df.head())

print("\nVegetables included:")
print(df["vegetable"].unique())
