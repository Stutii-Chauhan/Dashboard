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
        st.success(f"Successfully loaded `{uploaded_file.name}`")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if "df" in st.session_state:
    df = st.session_state.df

    # Data Preview
    st.subheader("Preview of the Data")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Custom Axis Selection
    st.markdown("---")
    st.subheader("Visualizations")
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    if categorical_cols:
        st.markdown("### Categorical Column Distributions")
        for col in categorical_cols:
            st.markdown(f"**{col}**")
            vc = df[col].value_counts().head(20)
            st.dataframe(vc)
            selected_x = st.selectbox(f"Select X-axis for {col} Bar Chart", options=[col], key=f"x_{col}")
            selected_y = st.selectbox(f"Select Y-axis for {col} Bar Chart", options=numeric_cols, key=f"y_{col}") if numeric_cols else None
            if selected_y:
                fig = px.bar(df, x=selected_x, y=selected_y, title=f"{selected_y} by {selected_x}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                vc = df[col].value_counts().head(20)
                fig = px.bar(x=vc.index, y=vc.values, labels={'x': col, 'y': 'Count'}, title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)

    if numeric_cols:
        st.markdown("### Histograms of Numeric Columns")
        for col in numeric_cols:
            selected_col = st.selectbox(f"Select numeric column for histogram", options=numeric_cols, index=numeric_cols.index(col), key=f"hist_{col}")
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
