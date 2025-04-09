from openai import OpenAI
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re
import math
import difflib
import numpy as np
from scipy import stats



#open ai key

openai.api_key = st.secrets["OPENAI_API_KEY"]

#defining open ai

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def query_openai(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful data analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM failed: {e}"

#detecting date time column

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
	
#Page name and layout

st.set_page_config(page_title="Data Analyzer", layout="wide")

# Theme Toggle with Switch
# Toggle stays stable, label doesn't change inside the toggle
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

# Toggle
col1, _ = st.columns([1, 9])
with col1:
    dark_mode = st.toggle("🌓", value=st.session_state.get("dark_mode", False), key="dark_mode")

# Show current mode label below (optional but pretty)
mode_icon = "🌙" if dark_mode else "🌞"
#mode_text = "Dark Mode" if dark_mode else "Light Mode"
#st.markdown(f"<span style='font-size: 14px;'>{mode_icon} {mode_text}</span>", unsafe_allow_html=True)

# Set the theme based on toggle
theme_mode = "Dark" if dark_mode else "Light"

# Inject custom CSS for themes
base_css = """
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background-color: {bg_color};
    color: {font_color};
}}

/* Inputs and Selectboxes */
input, textarea, select, [data-baseweb="input"], [data-baseweb="select"] {{
    background-color: {input_bg} !important;
    color: {font_color} !important;
    border: 1px solid #666 !important;
    border-radius: 5px !important;
}}

input::placeholder, textarea::placeholder {{
    color: {font_color}AA !important;
}}

div[data-baseweb="select"] > div {{
    color: {font_color} !important;
}}

/* Dropdown indicator */
svg {{
    fill: {font_color} !important;
}}

button, .stButton > button {{
    background-color: {button_bg} !important;
    color: {button_color} !important;
    border: none;
    padding: 0.4rem 1rem;
    border-radius: 5px;
}}

[data-testid="stFileDropzone"] {{
    background-color: {input_bg} !important;
    border: 1px dashed #999 !important;
    color: {font_color} !important;
}}

[data-testid="stFileDropzone"] * {{
    color: {font_color} !important;
}}

[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th,
.stCheckbox > label, .stRadio > div, label, p, h1, h2, h3, h4, h5, h6, span, div {{
    color: {font_color} !important;
}}
</style>
"""

# Apply updated theme styles
if theme_mode == "Dark":
    st.markdown(base_css.format(
        bg_color="#0e1117",
        font_color="#f1f1f1",
        input_bg="#1e1e1e",
        button_bg="#333333",
        button_color="#ffffff"
    ), unsafe_allow_html=True)
else:
    st.markdown(base_css.format(
        bg_color="#ffffff",
        font_color="#111111",
        input_bg="#f9f9f9",
        button_bg="#e1e1e1",
        button_color="#111111"
    ), unsafe_allow_html=True)
#Title and Subtitle

left_col, right_col = st.columns([1, 1])

with left_col:
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
	
	# Load data only once
	if uploaded_file is not None and "original_df" not in st.session_state:
	
	    try:
	        if uploaded_file.name.endswith(".csv"):
	            try:
	                df = pd.read_csv(uploaded_file, encoding='utf-8')
	            except UnicodeDecodeError:
	                try:
	                    df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
	                except Exception:
	                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
	        else:
	            df = pd.read_excel(uploaded_file)
	
	        df = df.reset_index(drop=True)
	        st.session_state.original_df = df  #  Keep raw
	        st.session_state.df = df.copy()    #  Working version
	        st.session_state.apply_header = False
	        st.success(f"Successfully loaded `{uploaded_file.name}`")
	    except Exception as e:
	        st.error(f"Error loading file: {e}")
	        st.stop()
	
	# CSV Tip
	if "df" in st.session_state:
	    st.markdown(
	    """
	    <span style='font-size: 13px;'> Tip: Save as <b>CSV UTF-8 (Comma delimited)</b></span>
	    """,
	    unsafe_allow_html=True
	)
	    # Apply header (only once per toggle)
	    apply_header = st.checkbox("Use first row as header", value=st.session_state.apply_header)
	
	    if apply_header != st.session_state.apply_header:
	        # Header was toggled
	        df = st.session_state.original_df.copy()
	
	        if apply_header:
	            new_header = df.iloc[0]
	            df = df[1:].copy()
	            df.columns = new_header
	        st.session_state.df = df
	        st.session_state.apply_header = apply_header
	
	    df = st.session_state.df  # Get working copy
	
	    # Convert datetime columns
	    datetime_cols = detect_datetime_columns(df)
	    for col in datetime_cols:
	        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
	    st.session_state.df = df
	
	    # Data preview
	    st.subheader("Preview of the Data")
	    st.dataframe(df.head(50))
	    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
	
	
	    # Column Classification
	    numeric_cols = list(df.select_dtypes(include='number').columns)
	    categorical_cols = [col for col in df.columns if col not in numeric_cols]
	
	    if (numeric_cols or categorical_cols) and st.checkbox("Show Dataset Overview"):
	        st.subheader("Dataset Overview")
	
	        if numeric_cols:
	            st.markdown("**Quantitative columns:**")
	            for col in numeric_cols:
	                st.write(f"- {col}")
	
	        if categorical_cols:
	            st.markdown("**Qualitative columns:**")
	            for col in categorical_cols:
	                st.write(f"- {col}")
	
	    if has_missing_data(df) and st.checkbox("Show Missing Value Handler"):
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
	
	    if numeric_cols and st.checkbox("Show Descriptive Statistics"):
	        st.subheader("Descriptive Statistics")
	        st.dataframe(df[numeric_cols].describe())
	
	    st.markdown("---")
	
	

	
	# --- Ask a Question Functionality (extended for missing values insight) ---
# --- Ask a Question Functionality (extended for missing values insight) ---
if "df" in st.session_state:
    df = st.session_state.df

    st.subheader("Ask a Question About Your Data")
    user_question = st.text_input("What do you want to know?")

    if user_question:
        q = user_question.lower()

        if "missing" in q:
            if "which column" in q and ("most" in q or "maximum" in q):
                missing_per_column = df.isna().sum()
                most_missing_col = missing_per_column.idxmax()
                count = missing_per_column.max()
                st.success(f"Column with the most missing values is '{most_missing_col}' with {count} missing entries.")
            elif "per column" in q or "column wise" in q or "each column" in q:
                missing_per_column = df.isna().sum()
                st.write("### Missing Values per Column")
                st.dataframe(missing_per_column[missing_per_column > 0])
            else:
                total_missing = df.isna().sum().sum()
                st.success(f"Total missing values in the dataset: {total_missing}")
                st.stop()

    if user_question:
        stat_keywords = {
            'mean': 'mean', 'average': 'mean', 'avg': 'mean', 'avrg': 'mean', 'av': 'mean', 'meanvalue': 'mean',
            'median': 'median', 'med': 'median',
            'mode': 'mode',
            'std': 'std', 'stdev': 'std', 'standard deviation': 'std',
            'variance': 'var', 'var': 'var',
            'min': 'min', 'minimum': 'min', 'lowest': 'min',
            'max': 'max', 'maximum': 'max', 'highest': 'max',
            'range': 'range',
            'iqr': 'iqr',
            'skew': 'skew',
            'kurtosis': 'kurtosis',
            '75%': '75th', '25%': '25th',
            'nulls': 'missing', 'missing': 'missing', 'nan': 'missing', 'na': 'missing', 'none': 'missing', 'blank': 'missing',
            '25th percentile': '25th', '75th percentile': '75th',
            'correlation': 'correlation', 'covariance': 'covariance',
            'regression': 'regression'
        }

        def get_column(col_candidate):
            col_candidate = col_candidate.strip().lower()
            col_candidate_clean = re.sub(r'[^a-z0-9 ]', '', col_candidate)
            cleaned_cols = {re.sub(r'[^a-z0-9 ]', '', col.lower()): col for col in df.columns}
            for cleaned, original in cleaned_cols.items():
                if col_candidate_clean in cleaned:
                    return original
            matches = difflib.get_close_matches(col_candidate_clean, cleaned_cols.keys(), n=1, cutoff=0.5)
            if matches:
                return cleaned_cols[matches[0]]
            return None

        # Handle correlation, regression, covariance
        if any(keyword in user_question.lower() for keyword in ['missing', 'null', 'nan', 'na', 'none', 'blank']):
            total_missing = df.isna().sum().sum()
            st.success(f"Total missing values in the dataset: {total_missing}")

        if any(keyword in user_question.lower() for keyword in ["correlation", "covariance", "regression"]):
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

        # Handle percentile and generic stats
        else:
            percentile_match = re.match(r".*?(\d{1,3})%.*?(?:of)?\s*([a-zA-Z0-9 _%()\-]+)", user_question, re.IGNORECASE)
            if percentile_match:
                perc, col_candidate = percentile_match.groups()
                perc = float(perc)
                col = get_column(col_candidate.strip().lower())
                if col and col in df.select_dtypes(include='number').columns:
                    try:
                        result = np.percentile(df[col].dropna(), perc)
                        st.success(f"The {perc}th percentile of {col} is {result:.4f}.")
                    except Exception as e:
                        st.error(f"Error while computing percentile: {e}")
                else:
                    st.warning("Could not match the column for your question.")
            else:
                stat_match = re.match(
                    r".*?(mean|average|avg|avrg|av|meanvalue|median|med|mode|std|stdev|standard deviation|variance|min|minimum|lowest|max|maximum|highest|range|iqr|skew|kurtosis).*?(?:of|for)?\s*([a-zA-Z0-9 _%()\-]+).*",
                    user_question, re.IGNORECASE)
                if stat_match:
                    stat, col_candidate = stat_match.groups()
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
                                result = df[col].describe().loc['25%']
                            elif stat_key == '75th':
                                result = df[col].describe().loc['75%']
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
                    st.info("Couldn't match to a known operation. Let me ask OpenAI.")
                    with st.spinner("Thinking..."):
                        try:
                            sample = df.head(10).to_csv(index=False)
                            prompt = f"""The user asked: '{user_question}'\n\nHere is a sample of the dataset:\n{sample}\n\nPlease provide a helpful and relevant answer based on this data."""
                            answer = query_openai(prompt)
                            st.success(answer)
                        except Exception as e:
                            st.error(f"Something went wrong with OpenAI: {e}")



# --- CUSTOM VISUALIZATION SECTION ---

# with right_col:
#     if "df" in st.session_state:
#         df = st.session_state.df
#         st.subheader("Create Your Own Chart")

# Only show chart builder if data is loaded
with right_col:
    if "df" in st.session_state:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols = df.columns.tolist()

        st.subheader("Create Your Own Chart")

        chart_type = st.selectbox("Choose chart type", [
            "Bar", "Column", "Pie", "Histogram",
            "Line", "Scatter", "Box",
            "Scatter with Regression", "Trendline (LOWESS)",
            "Correlation Heatmap"
        ])

        x_col = y_col = None
        fig = None

        # Common axis selectors
        if chart_type in ["Bar", "Column", "Line", "Scatter", "Box", "Scatter with Regression", "Trendline (LOWESS)", "Histogram"]:
            x_col = st.selectbox("Select X-axis", all_cols)

        if chart_type in ["Line", "Scatter", "Box", "Scatter with Regression", "Trendline (LOWESS)"]:
            y_options = [col for col in numeric_cols if col != x_col]
            if y_options:
                y_col = st.selectbox("Select Y-axis", y_options)
            else:
                y_col = None
                st.warning("No available numeric column for Y-axis.")

        if chart_type == "Pie":
            x_col = st.selectbox("Select category column for pie chart", all_cols)

        try:
            # Bar / Column Charts (both modes)
            if chart_type in ["Bar", "Column"] and x_col:
                bar_mode = st.radio("How do you want to build this chart?", ["Auto Count", "Custom X and Y"], horizontal=True)

                if bar_mode == "Auto Count":
                    value_counts = df[x_col].dropna().value_counts()
                    fig = px.bar(
                        x=value_counts.index if chart_type == "Column" else value_counts.values,
                        y=value_counts.values if chart_type == "Column" else value_counts.index,
                        orientation='v' if chart_type == "Column" else 'h',
                        labels={"x": x_col, "y": "Count"} if chart_type == "Column" else {"y": x_col, "x": "Count"}
                    )
                elif bar_mode == "Custom X and Y":
                    y_options = [col for col in numeric_cols if col != x_col]
                    if y_options:
                        y_col = st.selectbox("Select Y-axis (numeric)", y_options)
                        fig = px.bar(
                            df,
                            x=x_col if chart_type == "Column" else y_col,
                            y=y_col if chart_type == "Column" else x_col,
                            orientation='v' if chart_type == "Column" else 'h'
                        )
                    else:
                        st.warning("No valid numeric column available for Y-axis.")

            elif chart_type == "Histogram" and x_col:
                fig = px.histogram(df, x=x_col)

            elif chart_type == "Pie" and x_col:
                pie_vals = df[x_col].dropna().value_counts()
                fig = px.pie(names=pie_vals.index, values=pie_vals.values)

            elif chart_type == "Line" and x_col and y_col:
                fig = px.line(df, x=x_col, y=y_col)

            elif chart_type == "Scatter" and x_col and y_col:
                fig = px.scatter(df, x=x_col, y=y_col)

            elif chart_type == "Box" and x_col and y_col:
                fig = px.box(df, x=x_col, y=y_col)

            elif chart_type == "Scatter with Regression" and x_col and y_col:
                df_clean = df[[x_col, y_col]].dropna()
                fig = px.scatter(df_clean, x=x_col, y=y_col, trendline="ols")

            elif chart_type == "Trendline (LOWESS)" and x_col and y_col:
                df_clean = df[[x_col, y_col]].dropna()
                fig = px.scatter(df_clean, x=x_col, y=y_col, trendline="lowess")

            elif chart_type == "Correlation Heatmap" and numeric_cols:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type not in ["Correlation Heatmap"]:
                st.info("Please select appropriate columns to generate the chart.")

        except Exception as e:
            st.error(f"Error generating chart: {e}")
