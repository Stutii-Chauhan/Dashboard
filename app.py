# Streamlit Intelligent Dashboard (Clean Refactor)

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import difflib
from scipy import stats

# --- App Config ---
st.set_page_config(page_title="Data Analyzer", layout="wide")

# --- Theme Selector ---
with st.sidebar:
    theme_mode = st.radio("Choose Theme", ["Light", "Dark"], index=0)

# --- Dynamic CSS Styling ---
base_css = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {bg_color};
    color: {font_color};
}}
[data-testid="stSidebar"] > div:first-child {{ width: 230px; }}
input, textarea {{ background-color: {input_bg}; color: {font_color}; }}
button, .stButton > button {{ background-color: {button_bg}; color: {button_color}; border: none; }}
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{ color: {font_color} !important; }}
</style>
"""
if theme_mode == "Dark":
    st.markdown(base_css.format(bg_color="#0e1117", font_color="#ffffff", input_bg="#262730", button_bg="#444", button_color="#ffffff"), unsafe_allow_html=True)
else:
    st.markdown(base_css.format(bg_color="#ffffff", font_color="#000000", input_bg="#f0f2f6", button_bg="#dddddd", button_color="#000000"), unsafe_allow_html=True)

# --- Title ---
st.title("Intelligent Data Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        if df.columns[0] == df.iloc[0][0]:
            df.columns = df.iloc[0]
            df = df[1:]

        df = df.reset_index(drop=True)
        st.session_state.df = df
        st.success(f"Successfully loaded {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

if "df" not in st.session_state:
    st.warning("Please upload a dataset to begin.")
    st.stop()

# --- Detect datetime columns ---
def detect_datetime_columns(df):
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                if converted.notna().sum() > 0:
                    datetime_cols.append(col)
            except:
                continue
    return datetime_cols

# --- Main DataFrame ---
df = st.session_state.df

# Convert datetime columns
for col in detect_datetime_columns(df):
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
st.session_state.df = df

# --- Data Preview ---
st.subheader("🔍 Data Preview")
st.dataframe(df.head(50))
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# --- Dataset Overview ---
numeric_cols = list(df.select_dtypes(include='number').columns)
categorical_cols = [col for col in df.columns if col not in numeric_cols]

if st.checkbox("Show Dataset Overview"):
    st.subheader("📊 Dataset Overview")
    if numeric_cols:
        st.markdown("**Numeric columns:**")
        for col in numeric_cols:
            st.write(f"- {col}")
    if categorical_cols:
        st.markdown("**Categorical columns:**")
        for col in categorical_cols:
            st.write(f"- {col}")

# --- Missing Value Handler ---
missing_count = df.isna().sum().sum()
if missing_count > 0:
    with st.expander(f"🧹 Handle Missing Values ({int(missing_count)} missing)", expanded=True):

        st.markdown("Choose a strategy for missing values:")
        method = st.radio("Fill Method", ["Custom value", "Mean", "Median", "Mode"], horizontal=True)

        if method == "Custom value":
            custom_val = st.text_input("Enter custom fill value")

        col_scope = st.radio("Apply to", ["All columns", "Specific column"], horizontal=True)

        if col_scope == "Specific column":
            col_selected = st.selectbox("Select a column", df.columns[df.isna().any()])

        if st.button("Apply Fill"):
            if col_scope == "All columns":
                for col in df.columns:
                    if df[col].isna().any():
                        try:
                            if method == "Mean":
                                val = df[col].mean()
                            elif method == "Median":
                                val = df[col].median()
                            elif method == "Mode":
                                val = df[col].mode()[0]
                            else:
                                val = custom_val
                            df[col] = df[col].fillna(val)
                        except:
                            continue
            else:
                if method == "Mean":
                    val = df[col_selected].mean()
                elif method == "Median":
                    val = df[col_selected].median()
                elif method == "Mode":
                    val = df[col_selected].mode()[0]
                else:
                    val = custom_val
                df[col_selected] = df[col_selected].fillna(val)

            st.session_state.df = df
            st.success("Missing values filled successfully!")
            st.experimental_rerun()

    if st.button("🚫 Drop Rows with Missing Values"):
        original_shape = df.shape
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success(f"Dropped rows with missing values. Shape changed from {original_shape} to {df.shape}.")
        st.experimental_rerun()
else:
    st.success("✅ No missing values in your dataset!")
