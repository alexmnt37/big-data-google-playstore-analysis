# Google Play Store - Data Cleaning
import pandas as pd
import os

# First Step: Loading the Dataset

input_path = "data/raw/googleplaystore.csv"
output_path = "data/processed/googleplaystore_clean.csv"

df = pd.read_csv(input_path)

# Second Step: Initial inspection of the data

print("=" * 60)
print("INITIAL DATASET INSPECTION")
print("=" * 60)

print("\n--- Shape (rows, columns) ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

# Third Step: Finding the missing values

print("\n" + "=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
print(df.isnull().sum())

# 4th Step: Finding the duplicate information

print("\n" + "=" * 60)
print("DUPLICATE INFORMATION")
print("=" * 60)

exact_duplicates = df.duplicated().sum()
print(f"\nExact duplicate rows: {exact_duplicates}")

duplicate_app_names = df.duplicated(subset="App").sum()
print(f"Duplicate app names (keeping first): {duplicate_app_names}")

# 5th Step: Removing invalid rows

rows_before = len(df)

# Removing rows where Rating is greater than 5 (not a valid rating)
# Keep NaN ratings here — they will be handled later
df = df[(df["Rating"].isna()) | (df["Rating"] <= 5)]

rows_removed_invalid = rows_before - len(df)
print(f"\nRows removed (Rating > 5): {rows_removed_invalid}")

# 6th Step: Cleaning the reviews column

# Some values like "3.0M" are not valid integers — converting them safely
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

# 7th Step: Cleaning the installs column

# Removing commas and plus signs, then converting to numeric
# Example: "1,000,000+" → "1000000" → 1000000
df["Installs"] = df["Installs"].str.replace(",", "", regex=False)
df["Installs"] = df["Installs"].str.replace("+", "", regex=False)
df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

# 8th Step: Cleaning price column

# Removing the dollar sign, then converting to numeric
# Example: "$2.99" → "2.99" → 2.99
df["Price"] = df["Price"].str.replace("$", "", regex=False)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# 9th Step: Creating the SIZE_MB column

# Rules:
#  Values ending with M → strip M, keep as float (already in MB)
#  Values ending with k → strip k, convert to MB by dividing by 1024
# "Varies with device" → NaN (missing)

def convert_size_to_mb(size_value):
    """Convert app size string to a float value in MB."""
    if pd.isnull(size_value):
        return None
    size_value = str(size_value).strip()
    if size_value == "Varies with device":
        return None
    if size_value.endswith("M"):
        try:
            return float(size_value[:-1])
        except ValueError:
            return None
    if size_value.endswith("k"):
        try:
            return float(size_value[:-1]) / 1024
        except ValueError:
            return None
    return None

df["Size_MB"] = df["Size"].apply(convert_size_to_mb)

# 10th Step: Converting last updated to datetime

df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")

# 11th Step: Handling missing values

# Dropping rows where Rating is missing (target variable — cannot impute)
df = df.dropna(subset=["Rating"])

# Filling missing categorical columns with "Unknown"
df["Type"] = df["Type"].fillna("Unknown")
df["Content Rating"] = df["Content Rating"].fillna("Unknown")
df["Current Ver"] = df["Current Ver"].fillna("Unknown")
df["Android Ver"] = df["Android Ver"].fillna("Unknown")

# Filling missing Size_MB with the median (avoids skewing from outliers)
size_median = df["Size_MB"].median()
df["Size_MB"] = df["Size_MB"].fillna(size_median)

# 12th Step: Removing exact duplicate rows

rows_before_exact_dedup = len(df)
df = df.drop_duplicates()
rows_removed_exact = rows_before_exact_dedup - len(df)
print(f"Exact duplicate rows removed: {rows_removed_exact}")

# 13th Step: Removing duplicate apps — keeping the most recently updated

rows_before_app_dedup = len(df)

# Sorting by Last Updated descending so the most recent entry comes first
df = df.sort_values("Last Updated", ascending=False)

# Keeping the first occurrence of each app name (most recent after sorting)
df = df.drop_duplicates(subset="App", keep="first")

rows_removed_app_dedup = rows_before_app_dedup - len(df)
print(f"Duplicate app rows removed (kept most recent): {rows_removed_app_dedup}")

# Reset of the index after all removals
df = df.reset_index(drop=True)

# 14th Step: Saving the clean dataset

os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")

# 15th Step: final summary 

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print(f"\nOriginal row count : {rows_before}")
print(f"Final row count    : {len(df)}")
print(f"Total rows removed : {rows_before - len(df)}")

print("\n--- Final Shape ---")
print(df.shape)

print("\n--- Final Missing Values ---")
print(df.isnull().sum())

print("\n--- Final Data Types ---")
print(df.dtypes)

print("\n--- Cleaning Steps Performed ---")
print("  1. Removed rows with Rating > 5 (corrupt entries)")
print("  2. Converted Reviews to numeric (invalid values → NaN)")
print("  3. Stripped commas and '+' from Installs, converted to numeric")
print("  4. Stripped '$' from Price, converted to numeric")
print("  5. Created Size_MB column (M → float, k → /1024, Varies → NaN)")
print("  6. Converted Last Updated to datetime")
print("  7. Dropped rows with missing Rating")
print("  8. Filled missing Type, Content Rating, Ver columns with 'Unknown'")
print("  9. Filled missing Size_MB with median value")
print(" 10. Removed exact duplicate rows")
print(" 11. Removed duplicate app names — kept most recently updated version")