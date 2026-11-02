
import pandas as pd

# Load the dataset
df = pd.read_csv("datasets/vegetable_sales.csv")

# Convert date column into datetime format
df["date"] = pd.to_datetime(df["date"])

print("\n========== DATASET OVERVIEW ==========")

# Display dataset dimensions
print("Total rows and columns:", df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display first five records
print("\nFirst five records:")
print(df.head())

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicates:", df.duplicated().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# Average sales for each vegetable
print("\n========== AVERAGE VEGETABLE SALES ==========")

average_sales = df.groupby("vegetable")[
    "quantity_sold_kg"
].mean()

print(average_sales.round(2))

# Average price for each vegetable
print("\n========== AVERAGE VEGETABLE PRICES ==========")

average_prices = df.groupby("vegetable")[
    "price_per_kg"
].mean()

print(average_prices.round(2))

print("\nDataset analysis completed successfully!")
