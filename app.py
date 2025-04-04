import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Analyzer", layout="wide")

st.title("Analysis Dashboard")
st.markdown("Upload your Excel or CSV file to analyze and explore your dataset instantly.")

uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

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

    if st.button("Generate Business-Oriented Summary (LLM-Ready Text)"):
        summary_prompt = []
        summary_prompt.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. Below is a business-relevant overview:")

        if 'Revenue ($M)' in df.columns:
            revenue_mean = df['Revenue ($M)'].mean()
            summary_prompt.append(f"\n💰 **Revenue Insight**: The average revenue is {revenue_mean:.2f}M. Check for regional or seasonal variations.")

        if 'Profit Margin (%)' in df.columns:
            margin_avg = df['Profit Margin (%)'].mean()
            margin_min = df['Profit Margin (%)'].min()
            margin_max = df['Profit Margin (%)'].max()
            summary_prompt.append(f"\n📈 **Profit Margins**: Average = {margin_avg:.2f}%, Range = {margin_min:.2f}% to {margin_max:.2f}%.")

        if 'Operating Expenses ($M)' in df.columns:
            op_avg = df['Operating Expenses ($M)'].mean()
            summary_prompt.append(f"\n💸 **Operating Expenses**: The average expense is {op_avg:.2f}M. Compare with revenue trends to assess efficiency.")

        if 'Marketing Spend ($M)' in df.columns:
            mkt_spend = df['Marketing Spend ($M)'].mean()
            summary_prompt.append(f"\n📢 **Marketing Spend**: On average, {mkt_spend:.2f}M is allocated. Correlate with sales trends for ROI analysis.")

        if 'Employee Count (K)' in df.columns:
            emp_count = df['Employee Count (K)'].mean()
            summary_prompt.append(f"\n👥 **Employee Count**: Average team size is {emp_count:.2f}K. Analyze productivity metrics if available.")

        if 'R&D Investment ($M)' in df.columns:
            rnd = df['R&D Investment ($M)'].mean()
            summary_prompt.append(f"\n🔬 **R&D Spend**: Avg. R&D investment is {rnd:.2f}M. Evaluate impact on innovation/sales.")

        st.markdown("\n".join(summary_prompt))

    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    st.markdown("---")

    if (categorical_cols or numeric_cols) and st.checkbox("Show Basic Visualizations"):
        st.subheader("Basic Visualizations")

        if categorical_cols:
            st.markdown("### Categorical Column Distributions")
            for col in categorical_cols:
                st.markdown(f"**{col}**")
                vc = df[col].value_counts().head(20)
                st.dataframe(vc)
                fig = px.bar(x=vc.index, y=vc.values,
                             labels={'x': col, 'y': 'Count'},
                             title=f"{col} Distribution")
                st.plotly_chart(fig, use_container_width=True)

        if numeric_cols:
            st.markdown("### Histograms of Numeric Columns")
            for col in numeric_cols:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show Advanced Visualizations"):
        st.subheader("Advanced Visualizations")

        if categorical_cols:
            st.markdown("### Pie Charts (Top 5 Categories)")
            for col in categorical_cols:
                vc = df[col].value_counts().head(5)
                if len(vc) > 1:
                    fig = px.pie(values=vc.values, names=vc.index,
                                 title=f"{col} (Top 5 Categories)")
                    st.plotly_chart(fig, use_container_width=True)

        if numeric_cols:
            st.markdown("### Box Plots (Outlier Detection)")
            for col in numeric_cols:
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)

        if len(numeric_cols) > 1:
            st.markdown("### Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr,
                            text_auto=".2f",
                            title="Correlation Matrix",
                            aspect="auto",
                            color_continuous_scale="RdBu_r")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Scatter Plot (Select Variables)")
        if len(numeric_cols) >= 2:
            col1 = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            col2 = st.selectbox("Y-axis", [col for col in numeric_cols if col != col1], key="scatter_y")
            fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}")
            st.plotly_chart(fig, use_container_width=True)

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
