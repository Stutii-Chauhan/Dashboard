import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Successfully loaded `{uploaded_file.name}`")

        # Data Preview
        st.subheader("Preview of the Data")
        st.dataframe(df.head(50))
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Column Type Detection
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

        # Missing Values (merged display + handling options)
        total_missing = df.isna().sum().sum()
        if total_missing > 0:
            st.subheader("Missing Values")
            st.write(f"Total missing values: {int(total_missing)}")

            # Show rows with missing data
            missing_rows = df[df.isna().any(axis=1)]
            st.dataframe(missing_rows)

            st.markdown("---")
            st.subheader("Handle Missing Data")

            missing_cols = df.columns[df.isna().any()].tolist()

            # Option 1: Fill specific column
            st.markdown("**Fill a specific column with a custom value**")
            selected_col = st.selectbox("Choose a column", missing_cols, key="col_fill")
            fill_value = st.text_input(f"Enter value to fill in '{selected_col}':", key="val_fill")

            if fill_value:
                try:
                    dtype = df[selected_col].dropna().dtype
                    casted_value = dtype.type(fill_value)
                except:
                    casted_value = fill_value
                df[selected_col].fillna(casted_value, inplace=True)
                st.success(f"Filled missing values in '{selected_col}' with '{casted_value}'")

            st.markdown("---")

            # Option 2: Fill all missing values globally
            st.markdown("**Fill all missing values with a default**")
            default_fill = st.text_input("Enter a global default value:", key="fill_all")

            if default_fill:
                df.fillna(default_fill, inplace=True)
                st.success(f"All missing values filled with '{default_fill}'")

            st.markdown("---")

            # Option 3: Drop rows with any missing data
            if st.button("Drop all rows with missing values"):
                df.dropna(inplace=True)
                st.success("Dropped all rows containing missing values.")

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

    except Exception as e:
        st.error(f"Error reading file: {e}")
