import streamlit as st
import pandas as pd
import plotly.express as px
import sweetviz as sv
from streamlit.components.v1 import html
import tempfile

st.set_page_config(page_title="Intelligent Data Dashboard", layout="wide")

st.title("Intelligent Data Dashboard")
st.markdown("Upload an Excel or CSV file to get automated EDA insights and visualizations.")

# File Upload
uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load CSV or Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"File `{uploaded_file.name}` loaded successfully!")

        # Data Preview
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(50))
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Sweetviz EDA Report
        st.subheader("📊 Automated EDA Report (Sweetviz)")
        with st.spinner("Generating EDA report..."):
            report = sv.analyze(df)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                report.show_html(filepath=tmpfile.name, open_browser=False)
                st.success("EDA Report Ready!")
                html(open(tmpfile.name, "r", encoding="utf-8").read(), height=600, scrolling=True)

        # Example Chart - Top Category
        st.subheader("Example Chart: Top Categorical Column")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            top_col = cat_cols[0]
            chart_data = df[top_col].value_counts().reset_index()
            chart_data.columns = [top_col, "Count"]
            fig = px.bar(chart_data, x=top_col, y="Count", title=f"Top Values in '{top_col}'")
            st.plotly_chart(fig)
        else:
            st.info("No categorical columns found for charting.")

    except Exception as e:
        st.error(f"Error: {e}")
