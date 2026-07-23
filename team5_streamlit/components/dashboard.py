from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# =====================================================
# PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_DIR / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def dashboard(uploaded_file):

    st.markdown("""
    <h1 style='color:white;text-align:center;'>
    📊 SentiNova Dashboard
    </h1>
    """, unsafe_allow_html=True)

    uploaded_file.seek(0)   # Reset file pointer
    df = pd.read_csv(uploaded_file)

    review_column = "reviews.text"
    if "reviews.text" not in df.columns:
        st.error("The uploaded CSV must contain a 'reviews.text' column.")
        st.stop()

    reviews = df[review_column].fillna("").astype(str)

    tfidf = vectorizer.transform(reviews)

    predictions = model.predict(tfidf)

    df["Predicted Sentiment"] = predictions

    total = len(df)

    positive = (df["Predicted Sentiment"] == "Positive").sum()
    neutral = (df["Predicted Sentiment"] == "Neutral").sum()
    negative = (df["Predicted Sentiment"] == "Negative").sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", total)
    c2.metric("Positive", positive)
    c3.metric("Neutral", neutral)
    c4.metric("Negative", negative)

    # =====================================================
# SENTIMENT DISTRIBUTION
# =====================================================
    
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    sentiment_counts = df["Predicted Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    with left:

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count",
            hole=0.65,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#2ECC71",
                "Neutral": "#F1C40F",
                "Negative": "#E74C3C"
            }
        )

        fig.update_layout(

            title="🥧 Sentiment Distribution",

            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",

            font=dict(
                color="white",
                size=16
            ),

            legend=dict(
                orientation="h",
                y=-0.15
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        fig2 = px.bar(

            sentiment_counts,

            x="Sentiment",

            y="Count",

            color="Sentiment",

            color_discrete_map={

                "Positive":"#2ECC71",

                "Neutral":"#F1C40F",

                "Negative":"#E74C3C"

            }

        )

        fig2.update_layout(

            title="📊 Review Count",

            paper_bgcolor="#0E1117",

            plot_bgcolor="#0E1117",

            font=dict(color="white"),

            showlegend=False

        )

        st.plotly_chart(fig2,use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 AI Insights")

    total = len(df)

    positive_pct = (positive / total) * 100
    neutral_pct = (neutral / total) * 100
    negative_pct = (negative / total) * 100

    if positive_pct >= max(neutral_pct, negative_pct):
        overall = "Positive 😊"
    elif negative_pct >= max(positive_pct, neutral_pct):
        overall = "Negative 😟"
    else:
        overall = "Neutral 😐"

    st.info(f"""
    ### 📋 Summary

    • **Overall Customer Sentiment:** {overall}

    • Positive Reviews: **{positive_pct:.1f}%**

    • Neutral Reviews: **{neutral_pct:.1f}%**

    • Negative Reviews: **{negative_pct:.1f}%**

    ### 💡 Recommendation

    {"Customers are generally satisfied with the products." if positive_pct > 60 else
    "Customer opinions are mixed. Consider investigating the negative reviews." if negative_pct < 40 else
    "Many customers are dissatisfied. Product quality and customer experience should be reviewed."}
    """)


    st.markdown("---")
    st.subheader("📝 Review Predictions")

    st.dataframe(
        df[[review_column, "Predicted Sentiment"]],
        use_container_width=True,
        height=450
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="sentinova_predictions.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("✍️ Live Review Predictor")

    user_review = st.text_area(
        "Enter a product review:",
        placeholder="Example: The product quality is excellent and delivery was fast."
    )

    if st.button("🔍 Predict Sentiment"):

        if user_review.strip():

            review_vector = vectorizer.transform([user_review])

            prediction = model.predict(review_vector)[0]

            if prediction == "Positive":
                st.success(f"😊 Predicted Sentiment: {prediction}")

            elif prediction == "Neutral":
                st.warning(f"😐 Predicted Sentiment: {prediction}")

            else:
                st.error(f"😞 Predicted Sentiment: {prediction}")

        else:
            st.warning("Please enter a review first.")