# Google Play Store - Exploratory Data Analysis

import os
import pandas as pd
import matplotlib.pyplot as plt

# 1st Step: Loading the dataset

input_path = "data/processed/googleplaystore_clean.csv"
figures_path = "reports/figures"

os.makedirs(figures_path, exist_ok=True)

df = pd.read_csv(input_path)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns.tolist())

# Tracking all saved chart filenames
saved_figures = []

# 2nd Step: Chart 1 - Rating Distribution

print("\n" + "=" * 60)
print("CHART 1: Rating Distribution")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(df["Rating"].dropna(), bins=20, color="steelblue", edgecolor="white")

ax.set_title("Distribution of App Ratings", fontsize=14)
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Number of Apps", fontsize=12)

plt.tight_layout()

filename = "rating_distribution.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: Most apps are rated between 4.0 and 4.5, suggesting that")
print("         the Play Store skews toward positively rated applications.")

# 3rd Step: Chart 2 — Top 10 categories by number of apps

print("\n" + "=" * 60)
print("CHART 2: Top 10 Categories by Number of Apps")
print("=" * 60)

top_categories = (
    df["Category"]
    .value_counts()
    .head(10)
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(top_categories.index, top_categories.values, color="steelblue")

ax.set_title("Top 10 Categories by Number of Apps", fontsize=14)
ax.set_xlabel("Number of Apps", fontsize=12)
ax.set_ylabel("Category", fontsize=12)

plt.tight_layout()

filename = "top_categories_by_count.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: The Family and Game categories contain the most apps,")
print("         making them the most competitive segments on the store.")

# 4th Step: Chart 3 — Average rating by category

print("\n" + "=" * 60)
print("CHART 3: Average Rating by Category")
print("=" * 60)

avg_rating_by_category = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 10))

ax.barh(avg_rating_by_category.index, avg_rating_by_category.values, color="seagreen")

ax.set_title("Average Rating by Category", fontsize=14)
ax.set_xlabel("Average Rating", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
ax.set_xlim(0, 5)

plt.tight_layout()

filename = "avg_rating_by_category.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: Events and Education categories have the highest average ratings,")
print("         while Dating and Tools categories tend to receive lower scores.")

# 5th Step: Chart 4 — Free vs Paid distribution

print("\n" + "=" * 60)
print("CHART 4: Free vs Paid Distribution")
print("=" * 60)

type_counts = df["Type"].value_counts()

# Keep only Free and Paid for a clean pie chart
type_counts = type_counts[type_counts.index.isin(["Free", "Paid"])]

fig, ax = plt.subplots(figsize=(6, 6))

ax.pie(
    type_counts.values,
    labels=type_counts.index,
    autopct="%1.1f%%",
    colors=["steelblue", "salmon"],
    startangle=90
)

ax.set_title("Free vs Paid App Distribution", fontsize=14)

plt.tight_layout()

filename = "free_vs_paid.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: The vast majority of apps are free. This reflects the industry")
print("         standard of monetizing through ads and in-app purchases.")

# 6th Step: Chart 5 — Average rating: Free vs Paid

print("\n" + "=" * 60)
print("CHART 5: Average Rating — Free vs Paid")
print("=" * 60)

avg_rating_by_type = (
    df[df["Type"].isin(["Free", "Paid"])]
    .groupby("Type")["Rating"]
    .mean()
)

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(
    avg_rating_by_type.index,
    avg_rating_by_type.values,
    color=["steelblue", "salmon"],
    width=0.4
)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.02,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

ax.set_title("Average Rating: Free vs Paid Apps", fontsize=14)
ax.set_xlabel("App Type", fontsize=12)
ax.set_ylabel("Average Rating", fontsize=12)
ax.set_ylim(0, 5.5)

plt.tight_layout()

filename = "rating_free_vs_paid.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: Paid apps tend to have a slightly higher average rating than free apps.")
print("         Users who pay may have higher intent or developers invest more in quality.")

# 7th Step: Chart 6 — TOP 10 categories by average installs

print("\n" + "=" * 60)
print("CHART 6: Top 10 Categories by Average Installs")
print("=" * 60)

avg_installs_by_category = (
    df.groupby("Category")["Installs"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(avg_installs_by_category.index, avg_installs_by_category.values, color="darkorange")

ax.set_title("Top 10 Categories by Average Installs", fontsize=14)
ax.set_xlabel("Average Number of Installs", fontsize=12)
ax.set_ylabel("Category", fontsize=12)

# Formatting x-axis with readable numbers (millions)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.0f}M")
)

plt.tight_layout()

filename = "top_categories_by_installs.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: Communication and Social categories attract the most installs on average.")
print("         High install counts do not always align with the highest-rated categories.")

# 8th Step: Chart 7 — Content rating distribution

print("\n" + "=" * 60)
print("CHART 7: Content Rating Distribution")
print("=" * 60)

content_rating_counts = (
    df["Content Rating"]
    .value_counts()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(content_rating_counts.index, content_rating_counts.values, color="mediumpurple")

ax.set_title("Content Rating Distribution", fontsize=14)
ax.set_xlabel("Content Rating", fontsize=12)
ax.set_ylabel("Number of Apps", fontsize=12)

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

filename = "content_rating_distribution.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)
saved_figures.append(filename)

print(f"Saved: {filename}")
print("Insight: The majority of apps are rated 'Everyone', confirming that the")
print("         Play Store is primarily oriented toward general audiences.")

# 9th Step: Final Summary

print("\n" + "=" * 60)
print("EDA completed successfully")
print("=" * 60)

print("\nGenerated charts:")
for fname in saved_figures:
    print(f"  - reports/figures/{fname}")