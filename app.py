import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re
import math
import difflib
import numpy as np
from scipy import stats

st.set_page_config(page_title="Data Analyzer", layout="wide")

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

st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

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

        # Convert detected date columns to datetime
        datetime_cols = detect_datetime_columns(df)
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

        st.session_state.df = df  # Save for downstream use
        
        # Preview right after upload
        st.subheader("Preview of the Data")
        st.dataframe(df.head(50))
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    except Exception as e:
        st.error(f"Error loading file: {e}")



def has_missing_data(dataframe):
    return dataframe.isna().sum().sum() > 0

if "df" in st.session_state:
    df = st.session_state.df

    # Data Preview
    st.subheader("Preview of the Data")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Column Classification
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    if (numeric_cols or categorical_cols) and st.checkbox("Show Dataset Overview"):
        st.subheader("Dataset Overview")

        if numeric_cols:
            st.markdown("**Numeric columns:**")
            for col in numeric_cols:
                st.write(f"- {col}")

        if categorical_cols:
            st.markdown("**Categorical columns:**")
            for col in categorical_cols:
                st.write(f"- {col}")

    if has_missing_data(df) and st.checkbox("Show Missing Value Handler"):
        st.subheader("Missing Values")
        st.write(f"Total missing values: {int(df.isna().sum().sum())}")
        st.dataframe(df[df.isna().any(axis=1)])

        st.subheader("Handle Missing Data")
        missing_cols = df.columns[df.isna().any()].tolist()

        with st.expander("Fill a specific column", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_col = st.selectbox("Select column", missing_cols, key="col_fill")

            method = st.radio("How do you want to fill?", ["Custom value", "Mean", "Median", "Mode"], horizontal=True)

            fill_value = None

            if method == "Custom value":
                fill_input = st.text_input("Enter the value to fill:", key="custom_val")
                if fill_input:
                    try:
                        dtype = df[selected_col].dropna().dtype
                        fill_value = dtype.type(fill_input)
                    except:
                        fill_value = fill_input
            elif method == "Mean":
                fill_value = df[selected_col].mean()
            elif method == "Median":
                fill_value = df[selected_col].median()
            elif method == "Mode":
                mode_vals = df[selected_col].mode()
                fill_value = mode_vals[0] if not mode_vals.empty else None

            if st.button("Apply", key="apply_single_col"):
                if fill_value is not None:
                    df[selected_col].fillna(fill_value, inplace=True)
                    st.session_state.df = df
                    st.success(f"Filled missing values in '{selected_col}' using {method.lower()}: {fill_value}")
                    st.rerun()

        with st.expander("Fill all missing values (entire dataset)", expanded=False):
            fill_option = st.radio("Choose fill method", ["Custom value", "Mean", "Median", "Mode"], horizontal=True, key="fill_all_choice")

            if fill_option == "Custom value":
                global_default = st.text_input("Enter a global default value:", key="global_custom")
                if global_default and st.button("Apply Global Fill", key="fill_global_custom"):
                    df.fillna(global_default, inplace=True)
                    st.session_state.df = df
                    st.success(f"All missing values filled with '{global_default}'")
                    st.rerun()

            elif fill_option in ["Mean", "Median", "Mode"]:
                if st.button("Apply Global Fill", key="fill_global_stat"):
                    for col in df.columns:
                        if df[col].isna().any():
                            try:
                                if fill_option == "Mean":
                                    value = df[col].mean()
                                elif fill_option == "Median":
                                    value = df[col].median()
                                elif fill_option == "Mode":
                                    mode_vals = df[col].mode()
                                    value = mode_vals[0] if not mode_vals.empty else None
                                if value is not None:
                                    df[col].fillna(value, inplace=True)
                            except:
                                continue
                    st.session_state.df = df
                    st.success(f"Filled all missing values using column-wise {fill_option.lower()}")
                    st.rerun()

        with st.expander("Drop all rows with missing values", expanded=False):
            if st.button("Drop rows"):
                df.dropna(inplace=True)
                st.session_state.df = df
                st.success("Dropped all rows containing missing values.")
                st.rerun()

