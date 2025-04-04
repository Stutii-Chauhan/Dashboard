import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file you want to analyze")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Successfully loaded `{uploaded_file.name}`")
        
        st.subheader("Preview of the Data")
        st.dataframe(df.head(50))
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # ----------- PHASE 2: BASIC EDA STARTS HERE -----------

        st.subheader("Dataset Overview")

        # Detect column types
        numeric_cols = list(df.select_dtypes(include='number').columns)
        categorical_cols = [col for col in df.columns if col not in numeric_cols]

        st.markdown("**Numeric columns:**")
        for col in numeric_cols:
            st.write(f"- {col}")

        st.markdown("**Categorical columns:**")
        if categorical_cols:
            for col in categorical_cols:
                st.write(f"- {col}")
        else:
            st.write("None")
        

        # Missing Values
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if not missing.empty:
            st.subheader("🧼 Missing Values")
            st.write(f"Total missing values: **{int(missing.sum())}**")
            st.dataframe(missing)

            # Show rows with missing data
            st.subheader("🔎 Rows with Missing Data")
            missing_rows = df[df.isna().any(axis=1)]
            st.write(f"Showing {len(missing_rows)} rows with missing data:")
            st.dataframe(missing_rows)

        # Descriptive Stats
        if numeric_cols:
            st.subheader("Descriptive Statistics")
            st.dataframe(df[numeric_cols].describe())

        # Bar Charts for Categorical
        if categorical_cols:
            st.subheader("Categorical Column Distributions")
            for col in categorical_cols:
                st.markdown(f"**{col}**")
                vc = df[col].value_counts().head(20)
                st.dataframe(vc)
                fig = px.bar(x=vc.index, y=vc.values, labels={'x': col, 'y': 'Count'}, title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)

        # Histograms for Numeric
        if numeric_cols:
            st.subheader("Histograms of Numeric Columns")
            for col in numeric_cols:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

        # ----------- PHASE 2 ENDS HERE -----------

    except Exception as e:
        st.error(f"Error reading file: {e}")
