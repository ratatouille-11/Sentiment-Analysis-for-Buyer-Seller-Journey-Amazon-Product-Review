import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed_data/clean_amazon_reviews.csv")
X = df['clean_review']

# Output (Target)
y = df['Sentiment']

print(X.head())
print(y.head())

print(df['Sentiment'].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into numerical features
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(X)

print("Shape of TF-IDF matrix:", X.shape)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
"""
=============================================
    Training the Logistic Regression Model
===============================================
"""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained successfully!")

"""
==========================================
        Step 8: Make Predictions
==========================================
"""
y_pred = model.predict(X_test)

print(y_pred[:10])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

comparison = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted": y_pred
})

print(comparison.head(20))

"""
==========================================
        Test with Custom Reviews
==========================================
"""

reviews = [
    "This product is amazing and worth every penny.",
    "Very poor quality. Completely disappointed.",
    "The product is okay, nothing special."
]

reviews_tfidf = tfidf.transform(reviews)

predictions = model.predict(reviews_tfidf)

for review, sentiment in zip(reviews, predictions):
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment}\n")

import joblib

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("Model and TF-IDF vectorizer saved successfully!")