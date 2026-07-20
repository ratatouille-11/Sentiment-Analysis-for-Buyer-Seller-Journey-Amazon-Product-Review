"""
====================================================================
PROJECT : Sentiment Analysis for Buyer & Seller Journey (E-commerce)
TEAM 1  : Dataset -> Find (Kaggle: Amazon Product Reviews) + Understand + Clean
AUTHOR  : Saif
====================================================================

Dataset: Consumer Reviews of Amazon Products (Datafiniti / Kaggle)
File   : amazon_review_data.csv

This script:
  1. Loads the raw CSV and understands it (shape, dtypes, missing values)
  2. Drops unnecessary / empty columns
  3. Handles missing values in the important columns
  4. Removes duplicate records
  5. Cleans the review text (lowercase, punctuation, URLs, extra spaces)
  6. Creates the Sentiment column from reviews.rating
     (1-2 = Negative, 3 = Neutral, 4-5 = Positive)
  7. Verifies the final dataset is clean & ready for analysis
"""

import argparse
import re
import string

import pandas as pd


# ---------------------------------------------------------------
# STEP 1: Understand the raw dataset
# ---------------------------------------------------------------
def understand_dataset(df: pd.DataFrame) -> None:
    print("=" * 65)
    print("STEP 1: RAW DATASET OVERVIEW")
    print("=" * 65)
    print(f"Shape (rows, columns): {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print()


# ---------------------------------------------------------------
# STEP 2: Drop unnecessary columns
# ---------------------------------------------------------------
def select_important_columns(df: pd.DataFrame) -> pd.DataFrame:
    # The 11 "Unnamed: 10..20" columns are 100% empty -> junk from the raw export.
    # "categories" and "manufacturer" mostly duplicate "brand" for this dataset.
    keep_cols = [
        "name",               # product name
        "brand",               # closest available field to "seller" in this dataset
        "reviews.date",
        "reviews.rating",      # used to build Sentiment
        "reviews.text",        # the review itself
        "reviews.title",
        "reviews.doRecommend",
        "reviews.numHelpful",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    print("=" * 65)
    print("STEP 2: KEEPING IMPORTANT COLUMNS")
    print("=" * 65)
    dropped = set(df.columns) - set(keep_cols)
    print(f"Dropped columns ({len(dropped)}): {sorted(dropped)}")
    print(f"Kept columns    ({len(keep_cols)}): {keep_cols}\n")
    return df[keep_cols].copy()


# ---------------------------------------------------------------
# STEP 3: Missing values & duplicates
# ---------------------------------------------------------------
def clean_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 65)
    print("STEP 3: MISSING VALUES & DUPLICATES")
    print("=" * 65)
    before = len(df)

    # review text + rating are essential for sentiment analysis
    df = df.dropna(subset=["reviews.text", "reviews.rating"])
    print(f"Rows before: {before}  ->  after dropping missing review text/rating: {len(df)}")

    # fill less-critical missing fields instead of dropping more rows
    df["name"] = df["name"].fillna("Unknown Product")
    df["reviews.title"] = df["reviews.title"].fillna("")
    df["reviews.doRecommend"] = df["reviews.doRecommend"].fillna("Unknown")
    df["reviews.numHelpful"] = df["reviews.numHelpful"].fillna(0)
    df["reviews.date"] = df["reviews.date"].fillna("Unknown")

    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {dup_count}  ->  rows now: {len(df)}\n")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# STEP 4: Clean review text
# ---------------------------------------------------------------
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def clean_review_text(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 65)
    print("STEP 4: CLEANING REVIEW TEXT")
    print("=" * 65)
    df["clean_review"] = df["reviews.text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_review"].str.len() > 0].reset_index(drop=True)
    print(f"Rows dropped (empty after cleaning): {before - len(df)}")
    print("Example before -> after:")
    for i in range(3):
        print(f"  RAW  : {df.loc[i, 'reviews.text'][:120]}...")
        print(f"  CLEAN: {df.loc[i, 'clean_review'][:120]}...\n")
    return df


# ---------------------------------------------------------------
# STEP 5: Sentiment column from reviews.rating
# ---------------------------------------------------------------
def rating_to_sentiment(rating) -> str:
    rating = int(rating)
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 65)
    print("STEP 5: CREATING SENTIMENT LABEL")
    print("=" * 65)
    df["Sentiment"] = df["reviews.rating"].apply(rating_to_sentiment)
    print(df["Sentiment"].value_counts())
    print(f"\nSentiment distribution (%):")
    print((df["Sentiment"].value_counts(normalize=True) * 100).round(2))
    print()
    return df


# ---------------------------------------------------------------
# STEP 6: Final verification
# ---------------------------------------------------------------
def verify_clean_dataset(df: pd.DataFrame) -> None:
    print("=" * 65)
    print("STEP 6: FINAL VERIFICATION")
    print("=" * 65)
    print(f"Final shape         : {df.shape}")
    print(f"Remaining nulls     : {df.isnull().sum().sum()}")
    print(f"Remaining duplicates: {df.duplicated().sum()}")
    print(f"Sentiment classes   : {df['Sentiment'].unique().tolist()}")
    print(f"Unique products     : {df['name'].nunique()}")
    print(f"Unique brands       : {df['brand'].nunique()}")
    print("\nDataset is clean and ready for sentiment analysis!\n")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Clean Amazon reviews dataset for sentiment analysis")
    parser.add_argument("--input", required=True, help="Path to raw amazon_review_data CSV")
    parser.add_argument("--output", default="clean_amazon_reviews.csv", help="Path to save cleaned CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    understand_dataset(df)
    df = select_important_columns(df)
    df = clean_missing_and_duplicates(df)
    df = clean_review_text(df)
    df = add_sentiment_column(df)
    verify_clean_dataset(df)

    df.to_csv(args.output, index=False)
    print(f"[INFO] Cleaned dataset saved to: {args.output}")


if __name__ == "__main__":
    main()
