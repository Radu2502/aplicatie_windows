"""Aplicația ecvestră. Rulare: uv run --no-dev streamlit run app/app.py"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Postura călărețului", layout="wide")
PAGES = Path(__file__).parent / "pages"
pages = [
    st.Page(str(PAGES / "analiza.py"), title="Analiză", icon="📐", default=True),
]
st.navigation(pages).run()
