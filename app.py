import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Analyzer", layout="wide")

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

    # Phase 3 - Step 1: Prepare LLM-Friendly Summary
    if st.button("Prepare Data Summary for LLM"):
        summary_prompt = []

        # Dataset shape
        summary_prompt.append(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

        # Column info
        st.subheader("Column Details")
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            st.markdown(f"- **{col}**: Type = `{dtype}`, Missing = `{missing}`")

        # Numeric summary
        numeric_cols = df.select_dtypes(include='number').columns
        if not numeric_cols.empty:
            st.subheader("Summary of Numeric Columns")
            stats = df[numeric_cols].describe().T
            for col in stats.index:
                mean = stats.loc[col, 'mean']
                min_val = stats.loc[col, 'min']
                max_val = stats.loc[col, 'max']
                summary_prompt.append(f"{col}: Mean = {mean:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}")
                st.markdown(f"- **{col}**: Mean = `{mean:.2f}`, Min = `{min_val:.2f}`, Max = `{max_val:.2f}`")

        # Categorical summary
        cat_cols = df.select_dtypes(include='object').columns
        if not cat_cols.empty:
            st.subheader("Categorical Column Overview")
            for col in cat_cols:
                top_vals = df[col].value_counts().head(3).to_dict()
                summary_prompt.append(f"{col}: Top values = {top_vals}")
                st.markdown(f"- **{col}**: Top values = `{top_vals}`")

        # Save final prompt to session
        combined_prompt = "\n".join(summary_prompt)
        st.session_state.llm_prompt = combined_prompt
