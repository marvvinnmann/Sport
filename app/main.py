"""Streamlit entry point. Run with:  streamlit run app/main.py"""

import sys
from pathlib import Path

import streamlit as st

# make `import common` and the sportpred package work when launched from any folder
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parent / "src"))

st.set_page_config(page_title="sportpred — football value research", page_icon="⚽", layout="wide")

pages = [
    st.Page(str(APP_DIR / "pages" / "dashboard.py"), title="Dashboard", icon=":material/dashboard:", default=True),
    st.Page(str(APP_DIR / "pages" / "predictions.py"), title="Predictions", icon=":material/table_view:"),
    st.Page(str(APP_DIR / "pages" / "match_analysis.py"), title="Match Analysis", icon=":material/sports_soccer:"),
    st.Page(str(APP_DIR / "pages" / "model_performance.py"), title="Model Performance", icon=":material/monitoring:"),
    st.Page(str(APP_DIR / "pages" / "data_management.py"), title="Data Management", icon=":material/database:"),
    st.Page(str(APP_DIR / "pages" / "model_settings.py"), title="Model Settings", icon=":material/tune:"),
    st.Page(str(APP_DIR / "pages" / "backtesting.py"), title="Backtesting", icon=":material/history:"),
]
st.navigation(pages).run()
