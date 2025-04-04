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

    # Custom Graph Builder
    st.markdown("---")
    st.subheader("Build Your Own Graph")
    plot_type = st.selectbox("Select plot type", ["Bar", "Line", "Scatter", "Box", "Histogram", "Pie"])

    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = None
    if plot_type not in ["Pie", "Histogram"]:
        y_axis = st.selectbox("Select Y-axis", [col for col in df.columns if col != x_axis])

    if st.button("Generate Plot"):
        try:
            if plot_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis)
            elif plot_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis)
            elif plot_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis)
            elif plot_type == "Box":
                fig = px.box(df, x=x_axis, y=y_axis)
            elif plot_type == "Histogram":
                fig = px.histogram(df, x=x_axis)
            elif plot_type == "Pie":
                vc = df[x_axis].value_counts().reset_index()
                vc.columns = [x_axis, "Count"]
                fig = px.pie(vc, names=x_axis, values="Count")

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")
