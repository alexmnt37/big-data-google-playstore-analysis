# Google Play Store - PySpark Big Data Processing

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# 1st Step: Create the spark session

# SparkSession is gonna be the entry point to any PySpark program.
# It is run locally using all available CPU cores ("local[*]").

spark = SparkSession.builder \
    .appName("GooglePlayStoreAnalysis") \
    .master("local[*]") \
    .getOrCreate()

# Reducing verbose logging to keep console output readable
spark.sparkContext.setLogLevel("ERROR")

print("=" * 60)
print("PYSPARK BIG DATA PROCESSING")
print("=" * 60)
print("\nSparkSession created successfully.")

# 2nd Step: Loading the cleaned dataset

input_path = "data/processed/googleplaystore_clean.csv"

df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("multiLine", True) \
    .csv(input_path)

print(f"\nDataset loaded from: {input_path}")

# 3rd Step: Schema and basic inspection

print("\n--- Schema ---")
df.printSchema()

print("--- Total Number of Rows ---")
print(df.count())

print("\n--- First 5 Rows ---")
df.show(5, truncate=True)

# 4th Step: Category grouping
# Calculating number of apps, average rating, total installs, average reviews

print("\n" + "=" * 60)
print("CATEGORY SUMMARY: Apps, Rating, Installs, Reviews")
print("=" * 60)

category_summary = df.groupBy("Category").agg(
    F.count("App").alias("Number_of_Apps"),
    F.round(F.avg("Rating"), 2).alias("Avg_Rating"),
    F.sum("Installs").alias("Total_Installs"),
    F.round(F.avg("Reviews"), 0).alias("Avg_Reviews")
)

# Sorting by total installs descending
category_summary = category_summary.orderBy(F.col("Total_Installs").desc())

category_summary.show(truncate=False)

# 5th Step: Free vs paid apps comparison
# Comparing number of apps, average rating, and average installs

print("\n" + "=" * 60)
print("FREE VS PAID APP COMPARISON")
print("=" * 60)

free_vs_paid = df.filter(F.col("Type").isin(["Free", "Paid"])) \
    .groupBy("Type").agg(
        F.count("App").alias("Number_of_Apps"),
        F.round(F.avg("Rating"), 2).alias("Avg_Rating"),
        F.round(F.avg("Installs"), 0).alias("Avg_Installs")
    ) \
    .orderBy("Type")

free_vs_paid.show(truncate=False)

# 6th Step: Top 10 categories by average rating

print("\n" + "=" * 60)
print("TOP 10 CATEGORIES BY AVERAGE RATING")
print("=" * 60)

top_by_rating = df.groupBy("Category").agg(
    F.round(F.avg("Rating"), 2).alias("Avg_Rating"),
    F.count("App").alias("Number_of_Apps")
) \
    .orderBy(F.col("Avg_Rating").desc()) \
    .limit(10)

top_by_rating.show(truncate=False)

# 7th Step: Top 10 categories by total installs

print("\n" + "=" * 60)
print("TOP 10 CATEGORIES BY TOTAL INSTALLS")
print("=" * 60)

top_by_installs = df.groupBy("Category").agg(
    F.sum("Installs").alias("Total_Installs"),
    F.count("App").alias("Number_of_Apps")
) \
    .orderBy(F.col("Total_Installs").desc()) \
    .limit(10)

top_by_installs.show(truncate=False)

# 8th Step: Simple filtering examples

print("\n" + "=" * 60)
print("FILTERING: Apps with Rating >= 4.5")
print("=" * 60)

high_rated = df.filter(F.col("Rating") >= 4.5)

print(f"Number of apps with Rating >= 4.5: {high_rated.count()}")
high_rated.select("App", "Category", "Rating", "Installs") \
    .orderBy(F.col("Rating").desc()) \
    .show(10, truncate=True)

print("\n" + "=" * 60)
print("FILTERING: Apps with Installs >= 1,000,000")
print("=" * 60)

popular_apps = df.filter(F.col("Installs") >= 1_000_000)

print(f"Number of apps with Installs >= 1,000,000: {popular_apps.count()}")
popular_apps.select("App", "Category", "Rating", "Installs") \
    .orderBy(F.col("Installs").desc()) \
    .show(10, truncate=True)

# 10th Step: Stopping the sparksession

spark.stop()
print("SparkSession stopped. PySpark processing completed successfully.")