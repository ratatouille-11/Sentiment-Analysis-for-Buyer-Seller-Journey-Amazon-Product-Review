from pathlib import Path
from PIL import Image
import streamlit as st


# ===================================================
# PATHS
# ===================================================

BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

logo = Image.open(LOGO_PATH)


# ===================================================
# LANDING PAGE
# ===================================================

def landing():

    st.markdown("""
    <style>
    .block-container{
        padding-top:1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===========================
    # HEADER
    # ===========================

    left, right = st.columns([1, 6], gap="small")

    with left:
        st.image(logo, width=700)

    with right:

        st.markdown("""
        <div style="padding-top:10px">

        <h1 style="
        font-family:'Orbitron', sans-serif;
        font-size:68px;
        font-weight:800;
        letter-spacing:4px;
        margin-bottom:0;
        color:white;
        ">

        SENTI<span style="color:#D90429;">NOVA</span>

        <p style="
        font-size:22px;
        color:#D90429;
        margin-top:8px;
        font-weight:600;
        letter-spacing:1px;
        ">
        Decode. Discover. Decide.
        </p>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===========================
    # HERO TEXT
    # ===========================

    st.markdown("""

    <div style="text-align:center;">

    <h2 style="
    color:white;
    font-size:40px;
    margin-top:0px;
    margin-bottom:10px;
    ">Transform Reviews into Business Insights
    </h2>

    <p style="
    color:#aaaaaa;
    font-size:20px;
    width:80%;
    margin:auto;
    line-height:1.8;
    ">

    Upload your customer reviews and let
    <span style="color:#D90429;">SentiNova</span> uncover valuable business insights using AI.


    </p>

    </div>

    """, unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ===========================
    # FEATURE CARDS PLACEHOLDER
    # ===========================

    st.markdown("""
    <style>
    .feature-card{
        background:#161616;
        border:1px solid #2b2b2b;
        border-radius:18px;
        padding:25px;
        text-align:center;
        height:220px;
        padding:30px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
        transition:0.3s;
    }

    .feature-card:hover{
        border:1px solid #D90429;
        transform:translateY(-6px);
        box-shadow:0px 8px 25px rgba(217,4,41,0.25);
    }

    .feature-title{
        color:white;
        font-size:22px;
        font-weight:700;
        margin-top:10px;
    }

    .feature-text{
       .feature-text{
        font-size:15px;
        line-height:1.6;
        margin-top:10px;
        color:#BFBFBF;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
        <h1>📊</h1>
        <div class="feature-title">Analytics</div>
        <div class="feature-text">
        Interactive dashboards with real-time sentiment statistics.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
        <h1>🧠</h1>
        <div class="feature-title">AI Insights</div>
        <div class="feature-text">
        Machine learning uncovers customer emotions and trends.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
        <h1>✍️</h1>
        <div class="feature-title">Live Review</div>
        <div class="feature-text">
        Predict sentiment instantly from any customer review.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ===========================
    # UPLOAD SECTION PLACEHOLDER
    # ===========================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
    background:#121212;
    padding:35px;
    border-radius:20px;
    border:1px solid #2b2b2b;
    text-align:center;
    ">

    <h2 style="color:white;margin-bottom:10px;">
    Upload Your Dataset
    </h2>

    <p style="color:#B8B8B8;font-size:18px;">
    Choose a CSV file and let SentiNova generate AI-powered sentiment insights.
    </p>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["csv"],
        label_visibility="collapsed"
    )

    analyze = False

    if uploaded_file is not None:
        analyze = st.button(
            "🚀 Analyze Dataset",
            use_container_width=True
        )

    return uploaded_file, analyze