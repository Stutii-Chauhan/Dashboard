#Libraries

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re
import math
import difflib
import numpy as np
from scipy import stats

#Page Name and Layout 
st.set_page_config(page_title="Data Analyzer", layout="wide")

# Theme Toggle
with st.sidebar:
    st.markdown("<div style='padding-left: 10px;'>", unsafe_allow_html=True)
    theme_mode = st.radio("Choose Theme", ["Light", "Dark"], index=0)
    st.markdown("</div>", unsafe_allow_html=True)

# Inject custom CSS for themes with auto-contrast fonts
base_css = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {bg_color};
    color: {font_color};
}}

section[data-testid="stSidebar"] > div:first-child {{
    width: 230px;
}}

input, textarea, .stTextInput > div > input {{
    background-color: {input_bg};
    color: {font_color};
    border: 1px solid #ccc;
}}

button, .stButton > button {{
    background-color: {button_bg};
    color: {button_color};
    border: none;
    padding: 0.4rem 1rem;
    border-radius: 4px;
}}

.stCheckbox > label, .stRadio > div, label, p, h1, h2, h3, h4, h5, h6, span, div {{
    color: {font_color} !important;
}}

[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
    color: {font_color} !important;
}}
</style>
"""

if theme_mode == "Dark":
    st.markdown(base_css.format(
        bg_color="#0e1117",
        font_color="#ffffff",
        input_bg="#262730",
        button_bg="#444",
        button_color="#ffffff"
    ), unsafe_allow_html=True)
else:
    st.markdown(base_css.format(
        bg_color="#ffffff",
        font_color="#000000",
        input_bg="#f0f2f6",
        button_bg="#dddddd",
        button_color="#000000"
    ), unsafe_allow_html=True)

#Title and Subtitle
st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

#Uploading file
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

#checking file type and note

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
		try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except:
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')						   
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Successfully loaded `{uploaded_file.name}`")

        st.markdown("""
        <span style='font-size: 13px;'>
        Tip: If you're uploading a CSV exported from Excel, please save it as <b>CSV UTF-8 (Comma delimited)</b> to ensure best compatibility.
        </span>
        """, unsafe_allow_html=True)

        if 'apply_header' not in st.session_state:
            st.session_state.apply_header = False

        apply_header = st.checkbox("Use first row as header (if not already)", value=st.session_state.apply_header)
        st.session_state.apply_header = apply_header

# Apply header fix if checked
        if apply_header:
            new_header = df.iloc[0]
            df = df[1:].copy()
            df.columns = new_header
