import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

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

    # Descriptive Statistics
    if numeric_cols:
        st.subheader("Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe())

    # -----------------------------------------
    # ADVANCED VISUALIZATIONS (Toggle First)
    # -----------------------------------------
    st.markdown("---")
    if st.checkbox("Show Advanced Visualizations"):
        st.subheader("Advanced Visualizations")

        # Pie Charts
        if categorical_cols:
            st.markdown("### Pie Charts (Top 5 Categories)")
            for col in categorical_cols:
                vc = df[col].value_counts().head(5)
                if len(vc) > 1:
                    fig = px.pie(values=vc.values, names=vc.index,
                                 title=f"{col} (Top 5 Categories)")
                    st.plotly_chart(fig, use_container_width=True)

        # Box Plots
        if numeric_cols:
            st.markdown("### Box Plots (Outlier Detection)")
            st.caption("This box plot shows the spread of your data: median, quartiles, and outliers. Dots represent individual data points.")
            for col in numeric_cols:
                fig = px.box(df, y=col, points='all', title=f"Box Plot of {col}",
                             labels={col: col},
                             hover_data=[col])
                fig.update_traces(marker=dict(size=6, opacity=0.6))
                fig.update_layout(
                    yaxis_title=col,
                    hoverlabel=dict(bgcolor="white", font_size=12)
                )
                st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        if len(numeric_cols) > 1:
            st.markdown("### Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr,
                            text_auto=".2f",
                            title="Correlation Matrix",
                            aspect="auto",
                            color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

        # Scatter Plot
        if len(numeric_cols) >= 2:
            st.markdown("### Scatter Plot (Select Variables)")
            col1 = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            col2 = st.selectbox("Y-axis", [col for col in numeric_cols if col != col1], key="scatter_y")
            fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
            st.plotly_chart(fig, use_container_width=True)

        # Line Plot for Time Series
        datetime_cols = detect_datetime_columns(df)

        if datetime_cols and numeric_cols:
            st.markdown("### Time Series (Line Plot)")
            time_col = st.selectbox("Select datetime column", datetime_cols, key="line_dt")
            metric_col = st.selectbox("Select numeric column to plot", numeric_cols, key="line_val")
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            valid_rows = df[[time_col, metric_col]].dropna()
            if not valid_rows.empty and pd.api.types.is_datetime64_any_dtype(df[time_col]):
                fig = px.line(valid_rows.sort_values(by=time_col), x=time_col, y=metric_col,
                              title=f"{metric_col} over time ({time_col})")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Selected time or metric column doesn't have valid data to plot.")
