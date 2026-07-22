import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# ==========================================================
# TITLE
# ==========================================================

st.title("🛒 Sentiment Analysis for Buyer & Seller Journey")

st.info(
    "Upload a CSV file containing a column named **reviews.text**. "
    "The application will predict whether each review is Positive, Neutral or Negative."
)

# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

# ==========================================================
# MAIN APPLICATION
# ==========================================================

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # Check required column
    if "reviews.text" not in df.columns:
        st.error("❌ The uploaded CSV must contain a column named 'reviews.text'")
        st.stop()

    # ======================================================
    # DATA PREVIEW
    # ======================================================

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ======================================================
    # PREDICTIONS
    # ======================================================

    X = tfidf.transform(df["reviews.text"])

    predictions = model.predict(X)

    df["Predicted Sentiment"] = predictions

    # ======================================================
    # KPI CALCULATIONS
    # ======================================================

    total_reviews = len(df)

    positive = (df["Predicted Sentiment"] == "Positive").sum()

    neutral = (df["Predicted Sentiment"] == "Neutral").sum()

    negative = (df["Predicted Sentiment"] == "Negative").sum()

    st.divider()

    # ======================================================
    # KPI CARDS
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Total Reviews", total_reviews)

    with col2:
        st.metric("🟢 Positive", positive)

    with col3:
        st.metric("🟡 Neutral", neutral)

    with col4:
        st.metric("🔴 Negative", negative)

    st.divider()

    # ======================================================
    # CHART DATA
    # ======================================================

    sentiment_counts = (
        df["Predicted Sentiment"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "Sentiment",
        "Count"
    ]

    sentiment_counts = sentiment_counts.sort_values(
        by="Count",
        ascending=False
    )

    # ======================================================
    # DONUT CHART
    # ======================================================

    donut = px.pie(
        sentiment_counts,
        names="Sentiment",
        values="Count",
        hole=0.55,
        color="Sentiment",
        color_discrete_map={
            "Positive": "#2ecc71",
            "Neutral": "#f1c40f",
            "Negative": "#e74c3c"
        }
    )

    donut.update_layout(
        title="Sentiment Distribution",
        legend_title="Sentiment"
    )

    # ======================================================
    # BAR CHART
    # ======================================================

    bar = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        text="Count",
        color_discrete_map={
            "Positive": "#2ecc71",
            "Neutral": "#f1c40f",
            "Negative": "#e74c3c"
        }
    )

    bar.update_layout(
        title="Sentiment Count",
        showlegend=False
    )

    # ======================================================
    # DISPLAY CHARTS SIDE BY SIDE
    # ======================================================

    chart1, chart2 = st.columns(2)

    with chart1:
        st.plotly_chart(
            donut,
            use_container_width=True
        )

    with chart2:
        st.plotly_chart(
            bar,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # FILTER RESULTS
    # ======================================================

    st.subheader("🔍 Filter Predictions")

    option = st.selectbox(
        "Select Sentiment",
        ["All", "Positive", "Neutral", "Negative"]
    )

    if option == "All":
        filtered_df = df
    else:
        filtered_df = df[
            df["Predicted Sentiment"] == option
        ]

    # ======================================================
    # SHOW RESULTS
    # ======================================================

    st.subheader("📊 Prediction Results")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    # ======================================================
    # DOWNLOAD BUTTON
    # ======================================================

    csv = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Predicted CSV",
        data=csv,
        file_name="predicted_reviews.csv",
        mime="text/csv"
    )

    st.success(
        f"✅ Prediction completed successfully! "
        f"{total_reviews} reviews were analyzed."
    )