import streamlit as st
import time

def show_loading():

    status = st.empty()
    progress = st.progress(0)

    steps = [
        ("📂 Reading dataset...", 15),
        ("🧹 Cleaning reviews...", 30),
        ("📝 Vectorizing using TF-IDF...", 50),
        ("🤖 Running Logistic Regression...", 70),
        ("🧠 Generating AI Insights...", 90),
        ("✅ Preparing Dashboard...", 100),
    ]

    for text, value in steps:
        status.markdown(
            f"""
            <h3 style='text-align:center;color:white;'>
            {text}
            </h3>
            """,
            unsafe_allow_html=True,
        )

        progress.progress(value)
        time.sleep(0.8)

    status.empty()
    progress.empty()