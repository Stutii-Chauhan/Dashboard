import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re
import math
import difflib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats


#detecting date time column

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
	
#Page name and layout

st.set_page_config(page_title="Data Analyzer", layout="wide")

# if "buzz_history" not in st.session_state:
#     st.session_state.buzz_history = []

# Theme Toggle with Switch
# Toggle stays stable, label doesn't change inside the toggle
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Toggle
col1, _ = st.columns([1, 9])
with col1:
    dark_mode = st.toggle("🌓", value=st.session_state.get("dark_mode", False), key="dark_mode")

# Show current mode label below (optional but pretty)
mode_icon = "🌙" if dark_mode else "🌞"
#mode_text = "Dark Mode" if dark_mode else "Light Mode"
#st.markdown(f"<span style='font-size: 14px;'>{mode_icon} {mode_text}</span>", unsafe_allow_html=True)

# Set the theme based on toggle
theme_mode = "Dark" if dark_mode else "Light"

# Inject custom CSS for themes
base_css = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {bg_color};
    color: {font_color};
}}

/* Inputs and Selectboxes */
input, textarea, select, [data-baseweb="input"], [data-baseweb="select"] {{
    background-color: {input_bg} !important;
    color: {font_color} !important;
    border: 1px solid #666 !important;
    border-radius: 5px !important;
}}

input::placeholder, textarea::placeholder {{
    color: {font_color}AA !important;
}}

div[data-baseweb="select"] > div {{
    color: {font_color} !important;
}}

/* Dropdown indicator */
svg {{
    fill: {font_color} !important;
}}

button, .stButton > button {{
    background-color: {button_bg} !important;
    color: {button_color} !important;
    border: none;
    padding: 0.4rem 1rem;
    border-radius: 5px;
}}

[data-testid="stFileDropzone"] {{
    background-color: {input_bg} !important;
    border: 1px dashed #999 !important;
    color: {font_color} !important;
}}

[data-testid="stFileDropzone"] * {{
    color: {font_color} !important;
}}

[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
.stCheckbox > label, .stRadio > div, label, p, h1, h2, h3, h4, h5, h6, span, div {{
    color: {font_color} !important;
}}
</style>
"""

# Apply updated theme styles
if theme_mode == "Dark":
    st.markdown(base_css.format(
        bg_color="#0e1117",
        font_color="#f1f1f1",
        input_bg="#1e1e1e",
        button_bg="#333333",
        button_color="#ffffff"
    ), unsafe_allow_html=True)
else:
    st.markdown(base_css.format(
        bg_color="#ffffff",
        font_color="#111111",
        input_bg="#f9f9f9",
        button_bg="#e1e1e1",
        button_color="#111111"
    ), unsafe_allow_html=True)
#Title and Subtitle

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
	st.title("Analysis Dashboard")
	st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")
	
	uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
	
	def has_missing_data(dataframe):
	    return dataframe.isna().sum().sum() > 0
	
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
	
	# Load data only once
	if uploaded_file is not None and "original_df" not in st.session_state:
	
	    try:
	        if uploaded_file.name.endswith(".csv"):
	            try:
	                df = pd.read_csv(uploaded_file, encoding='utf-8')
	            except UnicodeDecodeError:
	                try:
	                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
	                except Exception:
	                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
	        else:
	            df = pd.read_excel(uploaded_file)
	
	        df = df.reset_index(drop=True)
	        st.session_state.original_df = df  #  Keep raw
	        st.session_state.df = df.copy()    #  Working version
	        st.session_state.apply_header = False
	        st.success(f"Successfully loaded `{uploaded_file.name}`")
	    except Exception as e:
	        st.error(f"Error loading file: {e}")
	        st.stop()
	
	# CSV Tip
	if "df" in st.session_state:
	    st.markdown(
	    """
	    <span style='font-size: 13px;'> Tip: Save as <b>CSV UTF-8 (Comma delimited)</b></span>
	    """,
	    unsafe_allow_html=True
	)
	    # Apply header (only once per toggle)
	    apply_header = st.checkbox("Use first row as header", value=st.session_state.apply_header)
	
	    if apply_header != st.session_state.apply_header:
	        # Header was toggled
	        df = st.session_state.original_df.copy()
	
	        if apply_header:
	            new_header = df.iloc[0]
	            df = df[1:].copy()
	            df.columns = new_header
	        st.session_state.df = df
	        st.session_state.apply_header = apply_header
	
	    df = st.session_state.df  # Get working copy
	
	    # Convert datetime columns
	    datetime_cols = detect_datetime_columns(df)
	    for col in datetime_cols:
	        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
	    st.session_state.df = df
	
	    # Data preview
	    st.subheader("Preview of the Data")
	    st.dataframe(df.head(50), use_container_width=True)
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
	
	    if numeric_cols and st.checkbox("Show Descriptive Statistics"):
	        st.subheader("Descriptive Statistics")
	        st.dataframe(df[numeric_cols].describe())
	
	    st.markdown("---")
	
	
	def query_huggingface(prompt, api_token, model="tiiuae/falcon-7b-instruct"):
	    API_URL = f"https://api-inference.huggingface.co/models/{model}"
	    headers = {"Authorization": f"Bearer {api_token}"}
	    payload = {
	        "inputs": prompt,
	        "parameters": {
	            "max_new_tokens": 150,
	            "temperature": 0.7,
	            "top_p": 0.9,
	            "repetition_penalty": 1.1,
	        }
	    }
	    response = requests.post(API_URL, headers=headers, json=payload)
	    try:
	        return response.json()[0]["generated_text"]
	    except:
	        return "LLM failed to generate a response. Please try again."
	
	

# --- CUSTOM VISUALIZATION SECTION ---

# with right_col:
#     if "df" in st.session_state:
#         df = st.session_state.df
#         st.subheader("Create Your Own Chart")

# Only show chart builder if data is loaded
with right_col:
    if "df" in st.session_state:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # Custom styled container (no st.container)
        st.markdown("""
            <div style='
                background-color: #f7f7f9;
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 0 8px rgba(0,0,0,0.05);
            '>
        """, unsafe_allow_html=True)

        st.subheader("Create Your Own Chart")

        chart_type = st.selectbox("Choose chart type", [
            "Line", "Bar", "Scatter", "Histogram", "Box",
            "Pie", "Scatter with Regression", "Trendline", "Correlation Heatmap"
        ])

        x_col = y_col = None

        if chart_type in ["Line", "Bar", "Scatter", "Box", "Histogram", "Scatter with Regression", "Trendline"]:
            x_col = st.selectbox("Select X-axis", df.columns)

        if chart_type in ["Line", "Bar", "Scatter", "Box", "Scatter with Regression", "Trendline"]:
            y_col = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != x_col])

        if chart_type == "Pie":
            x_col = st.selectbox("Select category column for pie chart", df.columns)

        fig = None
        try:
            if chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_col)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_col)
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col)
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col)
            elif chart_type == "Box":
                fig = px.box(df, x=x_col, y=y_col)
            elif chart_type == "Pie":
                pie_vals = df[x_col].dropna().value_counts()
                fig = px.pie(names=pie_vals.index, values=pie_vals.values)
            elif chart_type == "Scatter with Regression":
                import statsmodels.api as sm
                df_clean = df[[x_col, y_col]].dropna()
                fig = px.scatter(df_clean, x=x_col, y=y_col, trendline="ols")
            elif chart_type == "Trendline":
                df_clean = df[[x_col, y_col]].dropna()
                fig = px.scatter(df_clean, x=x_col, y=y_col, trendline="lowess")
            elif chart_type == "Correlation Heatmap":
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')

            if fig:
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating chart: {e}")

        # Close custom style container
        st.markdown("</div>", unsafe_allow_html=True)


# ---------- Floating Buzz Assistant (Bottom-Left Functional Bot) ----------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages in Streamlit-native chat layout (Streamlit 1.25+)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input at bottom — this stays fixed & styled by Streamlit
user_query = st.chat_input("Ask Buzz...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("assistant").markdown("Buzz here! I heard: **" + user_query + "**")
