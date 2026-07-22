"""
==============================================================
PROJECT : Sentiment Analysis for Buyer & Seller Journey
MODEL   : Logistic Regression + TF-IDF
AUTHOR  : Romana Zehra
==============================================================
"""

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# STEP 1 : Load Clean Dataset
# ==========================================================

df = pd.read_csv("team1_data/processed_data/clean_amazon_reviews.csv")

# Input Feature
X = df["clean_review"]

# Target Variable
y = df["Sentiment"]

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)
print(df["Sentiment"].value_counts())
print()

# ==========================================================
# STEP 2 : Convert Text into TF-IDF Features
# ==========================================================

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),   # Unigrams + Bigrams
    min_df=2              # Ignore extremely rare words
)

X = tfidf.fit_transform(X)

print("TF-IDF Matrix Shape:", X.shape)
print()

# ==========================================================
# STEP 3 : Split Dataset
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)
print()

# ==========================================================
# STEP 4 : Train Logistic Regression Model
# ==========================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")
print()

# ==========================================================
# STEP 5 : Predict on Test Data
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# STEP 6 : Evaluation
# ==========================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Accuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# STEP 7 : Compare Predictions
# ==========================================================

comparison = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": y_pred
})

print("\nSample Predictions")
print(comparison.head(20))

# ==========================================================
# STEP 8 : Test Custom Reviews
# ==========================================================

print("\n")
print("=" * 60)
print("CUSTOM REVIEW TEST")
print("=" * 60)

reviews = [

    "This product is amazing and worth every penny.",

    "Very poor quality. Completely disappointed.",

    "The product is okay, nothing special.",

    "Not bad at all.",

    "Not good.",

    "Waste of money.",

    "Absolutely fantastic!",

    "Absolutely terrible!"

]

reviews_tfidf = tfidf.transform(reviews)

predictions = model.predict(reviews_tfidf)

for review, sentiment in zip(reviews, predictions):

    print("Review :", review)
    print("Prediction :", sentiment)
    print()

# ==========================================================
# STEP 9 : Save Model
# ==========================================================

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("=" * 60)
print("Model Saved Successfully!")
print("=" * 60)

# ==========================================================
# STEP 10 : Interactive Prediction
# ==========================================================

print("\nSentiment Prediction System")

while True:

    review = input("\nEnter a review (type 'exit' to quit): ")

    if review.lower() == "exit":
        print("\nThank you!")
        break

    review_tfidf = tfidf.transform([review])

    prediction = model.predict(review_tfidf)[0]

    print("Predicted Sentiment :", prediction)