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

#title and subtitle

st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                except Exception:
                    df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            df = pd.read_excel(uploaded_file)

        df = df.reset_index(drop=True)
        st.session_state.df = df
        st.success(f"Successfully loaded {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()



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

# Load into session state once
if uploaded_file is not None and "df" not in st.session_state:							 
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df_try = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
					
                df_try = pd.read_csv(uploaded_file, encoding='latin1')

            if df_try.columns[0] == df_try.iloc[0][0]:  # likely header got shifted
                df_try.columns = df_try.iloc[0]  # use first row as header
                df_try = df_try[1:]  # drop the first row
            st.session_state.df = df_try.reset_index(drop=True)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

        st.success(f"Successfully loaded {uploaded_file.name}")																   

    except Exception as e:
        st.error(f"Error loading file: {e}")
	    
if "df" in st.session_state:
    df = st.session_state.df

    # Data Preview
    st.subheader("Preview of the Data")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # LLM Summary Generator Button
								 

    if st.button("Generate Dataset Summary (LLM Ready Text)"):
        summary = []
        summary.append(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.\n")
        summary.append("Column-wise summary:")
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            summary.append(f"- **{col}**: Type = {dtype}, Missing = {missing}")

        numeric_cols = df.select_dtypes(include='number').columns
        if not numeric_cols.empty:
            desc = df[numeric_cols].describe().T
            summary.append("\nKey statistics:")
            for col in desc.index:
                mean = desc.loc[col, 'mean']
                std = desc.loc[col, 'std']
                min_val = desc.loc[col, 'min']
                max_val = desc.loc[col, 'max']
                summary.append(f"- {col}: Mean = {mean:.2f}, Std = {std:.2f}, Range = [{min_val:.2f}, {max_val:.2f}]")

        st.markdown("\n".join(summary))


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

    # if numeric_cols and st.checkbox("Show Descriptive Statistics"):
    #     st.subheader("Descriptive Statistics")
    #     st.dataframe(df[numeric_cols].describe())

    # st.markdown("---")

    # if (categorical_cols or numeric_cols) and st.checkbox("Show Basic Visualizations"):
    #     st.subheader("Basic Visualizations")

    #     if categorical_cols:
    #         st.markdown("### Categorical Column Distributions")
    #         for col in categorical_cols:
    #             st.markdown(f"**{col}**")
    #             vc = df[col].value_counts().head(20)
    #             st.dataframe(vc)
    #             fig = px.bar(x=vc.index, y=vc.values,
    #                          labels={'x': col, 'y': 'Count'},
    #                          title=f"{col} Distribution")
    #             st.plotly_chart(fig, use_container_width=True)

    #     if numeric_cols:
    #         st.markdown("### Histograms of Numeric Columns")
    #         for col in numeric_cols:
    #             fig = px.histogram(df, x=col, title=f"Distribution of {col}")
    #             st.plotly_chart(fig, use_container_width=True)

    # if st.checkbox("Show Advanced Visualizations"):
    #     st.subheader("Advanced Visualizations")

    #     if categorical_cols:
    #         st.markdown("### Pie Charts (Top 5 Categories)")
    #         for col in categorical_cols:
    #             vc = df[col].value_counts().head(5)
    #             if len(vc) > 1:
    #                 fig = px.pie(values=vc.values, names=vc.index,
    #                              title=f"{col} (Top 5 Categories)")
    #                 st.plotly_chart(fig, use_container_width=True)

    #     if numeric_cols:
    #         st.markdown("### Box Plots (Outlier Detection)")
    #         for col in numeric_cols:
    #             fig = px.box(df, y=col, title=f"Box Plot of {col}")
    #             st.plotly_chart(fig, use_container_width=True)

    #     if len(numeric_cols) > 1:
    #         st.markdown("### Correlation Heatmap")
    #         corr = df[numeric_cols].corr()
    #         fig = px.imshow(corr,
    #                         text_auto=".2f",
    #                         title="Correlation Matrix",
    #                         aspect="auto",
    #                         color_continuous_scale="RdBu_r")
    #         st.plotly_chart(fig, use_container_width=True)

    #     st.markdown("### Scatter Plot (Select Variables)")
    #     if len(numeric_cols) >= 2:
    #         col1 = st.selectbox("X-axis", numeric_cols, key="scatter_x")
    #         col2 = st.selectbox("Y-axis", [col for col in numeric_cols if col != col1], key="scatter_y")
    #         fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
    #         st.plotly_chart(fig, use_container_width=True)

    #     datetime_cols = detect_datetime_columns(df)

    #     if datetime_cols and numeric_cols:
    #         st.markdown("### Time Series (Line Plot)")
    #         time_col = st.selectbox("Select datetime column", datetime_cols, key="line_dt")
    #         metric_col = st.selectbox("Select numeric column to plot", numeric_cols, key="line_val")
    #         df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    #         valid_rows = df[[time_col, metric_col]].dropna()
    #         if not valid_rows.empty and pd.api.types.is_datetime64_any_dtype(df[time_col]):
    #             fig = px.line(valid_rows.sort_values(by=time_col), x=time_col, y=metric_col,
    #                           title=f"{metric_col} over time ({time_col})")
    #             st.plotly_chart(fig, use_container_width=True)
    #         else:
    #             st.info("Selected time or metric column doesn't have valid data to plot.")

