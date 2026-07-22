import pandas as pd
import csv

# ==========================================================
# STEP 1 : Load Original Clean Dataset
# ==========================================================

df = pd.read_csv(
    "team1_data/processed_data/clean_amazon_reviews.csv"
)

print("=" * 65)
print("ORIGINAL DATASET")
print("=" * 65)
print(df["Sentiment"].value_counts())
print()

# ==========================================================
# STEP 2 : Separate Reviews by Sentiment
# ==========================================================

positive = df[df["Sentiment"] == "Positive"]
neutral = df[df["Sentiment"] == "Neutral"]
negative = df[df["Sentiment"] == "Negative"]

# ==========================================================
# STEP 3 : Create Balanced Tableau Dataset
# ==========================================================

# Positive Reviews (Downsample)
positive = positive.sample(
    n=21275,
    random_state=42
)

# Neutral Reviews (Upsample)
neutral = neutral.sample(
    n=7052,
    replace=True,
    random_state=42
)

# Negative Reviews (Upsample)
negative = negative.sample(
    n=11673,
    replace=True,
    random_state=42
)

# ==========================================================
# STEP 4 : Merge All Reviews
# ==========================================================

tableau_df = pd.concat([
    positive,
    neutral,
    negative
])

# Shuffle Dataset
tableau_df = tableau_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================================
# STEP 5 : Verify New Distribution
# ==========================================================

print("=" * 65)
print("BALANCED TABLEAU DATASET")
print("=" * 65)
print(tableau_df["Sentiment"].value_counts())
print()

print("Percentage Distribution")
print(
    (tableau_df["Sentiment"].value_counts(normalize=True) * 100).round(2)
)
print()

# ==========================================================
# STEP 7 : Save Excel (Recommended for Tableau)
# ==========================================================

tableau_df.to_excel(
    "team4_dashboard/tableau_amazon_reviews.xlsx",
    index=False
)

# ==========================================================
# STEP 8 : Completion Message
# ==========================================================

print("=" * 65)
print("FILES CREATED SUCCESSFULLY")
print("=" * 65)

print("CSV File")
print("team4_dashboard/tableau_amazon_reviews.csv")

print()

print("Excel File")
print("team4_dashboard/tableau_amazon_reviews.xlsx")

print()

print("Use tableau_amazon_reviews.xlsx in Tableau.")
print("Do NOT use this dataset for Machine Learning.")