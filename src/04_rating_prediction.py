# Google Play Store - Rating Prediction (Binary Classification)
# Model: Rule-Based Threshold Classifier (Pandas + NumPy + Matplotlib)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1st Step: Loading the dataset

input_path = "data/processed/googleplaystore_clean.csv"
figures_path = "reports/figures"

os.makedirs(figures_path, exist_ok=True)

df = pd.read_csv(input_path)

print("=" * 60)
print("RATING PREDICTION — BINARY CLASSIFICATION")
print("Model: Rule-Based Threshold Classifier")
print("=" * 60)

print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# 2nd Step: Creating the target variable

# Goal: predict whether an app has a HIGH rating (>= 4.0) or LOW rating (< 4.0)
# High_Rating = 1 means the app is rated 4.0 or above
# High_Rating = 0 means the app is rated below 4.0

df["High_Rating"] = (df["Rating"] >= 4.0).astype(int)

print("\n--- Target Variable Distribution ---")
print(df["High_Rating"].value_counts())
print("\n  0 = Low Rating  (< 4.0)")
print("  1 = High Rating (>= 4.0)")

# SECTION 3: Selecting features and dropping missing values

# Reviews and Installs are gonna be the two predictive features.
# Both are numeric and already cleaned in 01_data_cleaning.py.

feature_columns = ["Reviews", "Installs"]

model_df = df[feature_columns + ["High_Rating"]].dropna()

print(f"\nRows available for modeling (after dropna): {len(model_df)}")

# 4th Step: Manual training / Test spliting (80% / 20%)
# Sample 80% of rows for training using a fixed random_state for reproducibility.
# The remaining 20% become the test set.

train_df = model_df.sample(frac=0.8, random_state=42)
test_df  = model_df.drop(train_df.index)

print("\n--- Train / Test Split ---")
print(f"  Training rows : {len(train_df)}")
print(f"  Test rows     : {len(test_df)}")

# 5th Step: Learning thresholds from the training set
# The model "learns" by calculating the median Reviews and median Installs
# from the training data only — the test set is never seen at this stage.
# This is how even simple models separate learning from evaluation.

median_reviews  = train_df["Reviews"].median()
median_installs = train_df["Installs"].median()

print("\n--- Thresholds Learned from Training Set ---")
print(f"  Median Reviews  : {median_reviews:,.0f}")
print(f"  Median Installs : {median_installs:,.0f}")

# 6th Step: Prediction rule establishment
# Rule:
#   If Reviews >= median_reviews OR Installs >= median_installs → predict High (1)
#   Otherwise → predict Low (0)
# Logic: apps that are widely reviewed OR widely installed are more likely
# to have an established user base that keeps ratings high.

print("\n--- Prediction Rule ---")
print("  IF Reviews >= median_reviews OR Installs >= median_installs")
print("      → predict High_Rating = 1")
print("  ELSE")
print("      → predict High_Rating = 0")

# Applying the rule to the test set
test_df = test_df.copy()

test_df["Predicted"] = np.where(
    (test_df["Reviews"] >= median_reviews) | (test_df["Installs"] >= median_installs),
    1,
    0
)

# 7th Step: Calculating the accuracy manually

correct_predictions = (test_df["Predicted"] == test_df["High_Rating"]).sum()
total_predictions   = len(test_df)
accuracy            = correct_predictions / total_predictions

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\n  Total test rows       : {total_predictions}")
print(f"  Correct predictions   : {correct_predictions}")
print(f"  Accuracy              : {accuracy:.4f}  ({accuracy * 100:.2f}%)")

# 8th Step: Calculating confusing matrix manually

# A confusion matrix shows 4 types of prediction outcomes:
#
#   True Negative  (TN): actual Low,  predicted Low  — correct
#   False Positive (FP): actual Low,  predicted High — wrong
#   False Negative (FN): actual High, predicted Low  — wrong
#   True Positive  (TP): actual High, predicted High — correct

TN = ((test_df["High_Rating"] == 0) & (test_df["Predicted"] == 0)).sum()
FP = ((test_df["High_Rating"] == 0) & (test_df["Predicted"] == 1)).sum()
FN = ((test_df["High_Rating"] == 1) & (test_df["Predicted"] == 0)).sum()
TP = ((test_df["High_Rating"] == 1) & (test_df["Predicted"] == 1)).sum()

print("\n--- Confusion Matrix Values ---")
print(f"  True Negatives  (Low  predicted as Low)  : {TN}")
print(f"  False Positives (Low  predicted as High) : {FP}")
print(f"  False Negatives (High predicted as Low)  : {FN}")
print(f"  True Positives  (High predicted as High) : {TP}")

# Building the 2x2 matrix as a NumPy array for plotting
cm = np.array([[TN, FP],
               [FN, TP]])

# 9th Step: Saving the confusion matrix figure

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)

# Axis labels
classes = ["Low Rating (0)", "High Rating (1)"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=10)
ax.set_yticklabels(classes, fontsize=10)

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix — Rule-Based Classifier", fontsize=13)

# Printing cell values inside each square
for i in range(2):
    for j in range(2):
        ax.text(
            j, i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14,
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.tight_layout()

filename = "confusion_matrix.png"
fig.savefig(os.path.join(figures_path, filename))
plt.close(fig)

print(f"\nConfusion matrix saved to: reports/figures/{filename}")

# 11th Step: Final summary

print("=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)
print(f"\n  Model             : Rule-Based Threshold Classifier")
print(f"  Target            : High_Rating (Rating >= 4.0 → 1, else → 0)")
print(f"  Features used     : Reviews, Installs")
print(f"  Threshold Reviews : {median_reviews:,.0f}")
print(f"  Threshold Installs: {median_installs:,.0f}")
print(f"  Training rows     : {len(train_df)}")
print(f"  Test rows         : {len(test_df)}")
print(f"  Accuracy          : {accuracy * 100:.2f}%")
print(f"\nRating prediction completed successfully.")