import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re
import difflib
import numpy as np
from scipy import stats

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

def query_huggingface(prompt, api_token, model="tiiuae/falcon-7b-instruct"):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except:
        return "LLM failed to generate a response. Please try again."

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

    st.subheader("Preview of the Data")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Ask a Question
    st.subheader("\U0001F9E0 Ask a Question About Your Data")
    user_question = st.text_input("What do you want to know?")

    if user_question:
        stat_keywords = {
            'mean': 'mean', 'average': 'mean', 'avg': 'mean', 'avrg': 'mean', 'av': 'mean', 'meanvalue': 'mean',
            'median': 'median', 'med': 'median',
            'mode': 'mode',
            'std': 'std', 'stdev': 'std', 'standard deviation': 'std',
            'variance': 'var',
            'min': 'min', 'minimum': 'min', 'lowest': 'min',
            'max': 'max', 'maximum': 'max', 'highest': 'max',
            'range': 'range',
            'iqr': 'iqr',
            'skew': 'skew',
            'kurtosis': 'kurtosis',
            '25th percentile': '25th', '75th percentile': '75th',
            'correlation': 'correlation', 'covariance': 'covariance',
            'regression': 'regression'
        }

        def get_column(col_candidate):
            possible_matches = [col for col in df.columns if col_candidate in col.lower()]
            if not possible_matches:
                possible_matches = difflib.get_close_matches(col_candidate, df.columns, n=1, cutoff=0.6)
            return possible_matches[0] if possible_matches else None

        if "correlation" in user_question.lower() or "covariance" in user_question.lower() or "regression" in user_question.lower():
            cols = re.findall(r"[a-zA-Z0-9 _%()\-]+", user_question)
            matched_cols = [get_column(c.lower()) for c in cols if get_column(c.lower()) in df.columns]
            if len(matched_cols) >= 2:
                col1, col2 = matched_cols[:2]
                if "correlation" in user_question.lower():
                    val = df[col1].corr(df[col2])
                    st.success(f"Correlation between {col1} and {col2} is {val:.4f}.")
                elif "covariance" in user_question.lower():
                    val = df[col1].cov(df[col2])
                    st.success(f"Covariance between {col1} and {col2} is {val:.4f}.")
                elif "regression" in user_question.lower():
                    result = stats.linregress(df[col1].dropna(), df[col2].dropna())
                    st.success(f"Regression between {col1} and {col2}: Slope = {result.slope:.4f}, Intercept = {result.intercept:.4f}, R = {result.rvalue:.4f}")
                else:
                    st.warning("Could not determine type of relationship analysis.")
            else:
                st.warning("Please mention two valid numeric columns.")
        else:
            pattern = r".*?(mean|average|avg|avrg|av|meanvalue|median|med|mode|std|stdev|standard deviation|variance|min|minimum|lowest|max|maximum|highest|range|iqr|skew|kurtosis|25th percentile|75th percentile).*?(?:of|for)?\s*([a-zA-Z0-9 _%()\-]+).*"
            match = re.match(pattern, user_question, re.IGNORECASE)
            if match:
                stat, col_candidate = match.groups()
                stat_key = stat_keywords.get(stat.lower(), None)
                col = get_column(col_candidate.strip().lower())
                if col and col in df.select_dtypes(include='number').columns:
                    try:
                        if stat_key == 'mean':
                            result = df[col].mean()
                        elif stat_key == 'median':
                            result = df[col].median()
                        elif stat_key == 'mode':
                            result = df[col].mode().iloc[0]
                        elif stat_key == 'std':
                            result = df[col].std()
                        elif stat_key == 'var':
                            result = df[col].var()
                        elif stat_key == 'min':
                            result = df[col].min()
                        elif stat_key == 'max':
                            result = df[col].max()
                        elif stat_key == 'range':
                            result = df[col].max() - df[col].min()
                        elif stat_key == 'iqr':
                            result = np.percentile(df[col].dropna(), 75) - np.percentile(df[col].dropna(), 25)
                        elif stat_key == 'skew':
                            result = df[col].skew()
                        elif stat_key == 'kurtosis':
                            result = df[col].kurtosis()
                        elif stat_key == '25th':
                            result = np.percentile(df[col].dropna(), 25)
                        elif stat_key == '75th':
                            result = np.percentile(df[col].dropna(), 75)
                        else:
                            result = None

                        if result is not None:
                            st.success(f"The {stat} of {col} is {result:.4f}.")
                        else:
                            st.warning("This operation is not supported yet.")
                    except Exception as e:
                        st.error(f"Error while computing: {e}")
                else:
                    st.warning("Could not match the column for your question.")
            else:
                st.info("Couldn't match to a known operation. Please rephrase or check column names.")



      # Column Classification
    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    if (numeric_cols or categorical_cols) and st.checkbox("Show Dataset Overview", key="dataset_overview_checkbox"):
        st.subheader("Dataset Overview")

        if numeric_cols:
            st.markdown("**Numeric columns:**")
            for col in numeric_cols:
                st.write(f"- {col}")

        if categorical_cols:
            st.markdown("**Categorical columns:**")
            for col in categorical_cols:
                st.write(f"- {col}")

    if has_missing_data(df) and st.checkbox("Show Missing Value Handler", key="missing_value_checkbox"):
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

    if numeric_cols and st.checkbox("Show Descriptive Statistics", key="descriptive_stats_checkbox"):
        st.subheader("Descriptive Statistics")
        st.dataframe(df[numeric_cols].describe())

    st.markdown("---")

    if (categorical_cols or numeric_cols) and st.checkbox("Show Basic Visualizations", key="basic_viz_checkbox"):
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

    if st.checkbox("Show Advanced Visualizations", key="advanced_viz_checkbox"):
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
