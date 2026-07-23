from pathlib import Path
import streamlit as st

from components.landing import landing
from components.loading import show_loading
from components.dashboard import dashboard

st.set_page_config(
    page_title="SentiNova",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

css_path = Path(__file__).parent / "style.css"

with open(css_path, encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "landing"

# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------

if st.session_state.page == "landing":

    uploaded_file, analyze = landing()

    if uploaded_file is not None and analyze:

        st.session_state.uploaded_file = uploaded_file

        show_loading()

        st.session_state.page = "dashboard"

        st.rerun()

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

elif st.session_state.page == "dashboard":

    dashboard(st.session_state.uploaded_file)