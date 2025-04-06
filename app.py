import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

def has_missing_data(dataframe):
    return dataframe.isna().sum().sum() > 0

# Load into session state once
if uploaded_file is not None and "df" not in st.session_state:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success(f"Successfully loaded `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if "df" in st.session_state:
    df = st.session_state.df

    # Data Preview
    st.subheader("Preview of the Data")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Column Classification
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    if numeric_cols or categorical_cols:
        st.subheader("Dataset Overview")

    if numeric_cols:
        st.markdown("**Numeric columns:**")
        for col in numeric_cols:
            st.write(f"- {col}")

    if categorical_cols:
        st.markdown("**Categorical columns:**")
        for col in categorical_cols:
            st.write(f"- {col}")

    # Missing Values
    if has_missing_data(df):
        st.subheader("Missing Values")
        st.write(f"Total missing values: {int(df.isna().sum().sum())}")
        st.dataframe(df[df.isna().any(axis=1)])

        st.subheader("Handle Missing Data")
        missing_cols = df.columns[df.isna().any()].tolist()

        # 1. Fill a specific column
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
                    st.experimental_rerun()

        # 2. Global fill for all missing values
        with st.expander("Fill all missing values (entire dataset)", expanded=False):
            fill_option = st.radio("Choose fill method", ["Custom value", "Mean", "Median", "Mode"], horizontal=True, key="fill_all_choice")

            if fill_option == "Custom value":
                global_default = st.text_input("Enter a global default value:", key="global_custom")
                if global_default and st.button("Apply Global Fill", key="fill_global_custom"):
                    df.fillna(global_default, inplace=True)
                    st.session_state.df = df
                    st.success(f"All missing values filled with '{global_default}'")
                    st.experimental_rerun()

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
                    st.experimental_rerun()

        # 3. Drop rows with missing values
        with st.expander("Drop all rows with missing values", expanded=False):
            if st.button("Drop rows"):
                df.dropna(inplace=True)
                st.session_state.df = df
                st.success("Dropped all rows containing missing values.")
                st.experimental_rerun()

    # Descriptive Statistics
    if numeric_cols:
        st.subheader("Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe())

    # Categorical Distributions
    if categorical_cols:
        st.subheader("Categorical Column Distributions")
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            vc = df[col].value_counts().head(20)
            st.dataframe(vc)
            fig = px.bar(x=vc.index, y=vc.values,
                         labels={'x': col, 'y': 'Count'},
                         title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Numeric Distributions
    if numeric_cols:
        st.subheader("Histograms of Numeric Columns")
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
