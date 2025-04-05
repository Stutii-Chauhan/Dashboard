import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re

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

    # Data Preview
    st.subheader("Preview of the Data")
    st.dataframe(df.head(50))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # LLM-Enhanced Insight Button
    if st.button("Generate Business Summary using AI"):
        numeric_cols = df.select_dtypes(include='number').columns
        desc = df[numeric_cols].describe().T

        metrics_summary = "\n\n".join(
            [
                f"Column: {col}\nMean: {desc.loc[col, 'mean']:.2f}\nStd: {desc.loc[col, 'std']:.2f}\nMin: {desc.loc[col, 'min']:.2f}\nMax: {desc.loc[col, 'max']:.2f}"
                for col in desc.index
            ]
        )

        prompt = (
            f"Answer the following based on the dataset:\n"
            f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}\n\n"
            f"Summary statistics:\n{metrics_summary}\n\n"
            f"Question: What trends and insights can you derive from this data?"
        )

        hf_token = st.secrets["hf_token"]
        with st.spinner("Generating AI business summary..."):
            response = query_huggingface(prompt, hf_token)

        cleaned_response = response.strip()
        lines = [line.strip() for line in cleaned_response.split("\n") if line.strip() and not line.strip().lower().startswith("as an ai")]
        final_output = lines[-1] if lines else "Summary could not be generated."

        st.subheader("\U0001F4A1 AI-Generated Business Summary")
        st.markdown(
            f"<div style='background-color:#f0f8f5; padding: 15px; border-radius: 8px; font-size: 15px; white-space: pre-wrap'>{final_output}</div>",
            unsafe_allow_html=True
        )

    # Ask a Question Section
    st.subheader("\U0001F9E0 Ask a Question About Your Data")
    user_question = st.text_input("What do you want to know?")
    if user_question:
        # Handle direct missing column question
        if "missing" in user_question.lower() and "column" in user_question.lower():
            missing_counts = df.isna().sum()
            missing_col = missing_counts[missing_counts > 0].idxmax()
            st.markdown(
                f"<div style='background-color:#f0f8f5; padding: 12px; border-radius: 6px; font-size: 15px;'>The column with missing values is <b>{missing_col}</b>.</div>",
                unsafe_allow_html=True
            )

        # Check for exact stat questions
        else:
            match = re.match(r".*(mean|average|median|max|min|std).*?(?:of|for)?\s*([a-zA-Z0-9 _%()-]+).*", user_question, re.IGNORECASE)
            if match:
                stat, col_candidate = match.groups()
                stat = stat.lower().strip()
                col_candidate = col_candidate.strip().lower()

                def best_column_match(candidate, df_columns):
                    for col in df_columns:
                        if candidate in col.lower():
                            return col
                    return None

                matched_col = best_column_match(col_candidate, df.columns)

                if matched_col and matched_col in df.select_dtypes(include='number').columns:
                    stat_map = {
                        'mean': 'mean',
                        'average': 'mean',
                        'median': '50%',
                        'max': 'max',
                        'min': 'min',
                        'std': 'std'
                    }
                    stat_key = stat_map.get(stat)
                    if stat_key:
                        value = df[matched_col].describe().get(stat_key)
                        if value is not None:
                            st.markdown(
                                f"<div style='background-color:#f0f8f5; padding: 12px; border-radius: 6px; font-size: 15px;'>The {stat} of '<b>{matched_col}</b>' is <b>{value:.4f}</b>.</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("Could not find the requested statistic.")
                    else:
                        st.warning("Unsupported statistic requested.")
                else:
                    st.warning("Could not match the column for your question.")
            else:
                # Fallback to LLM if it's not a clean stat question
                question_prompt = (
                    f"Answer the following based on the dataset:\n"
                    f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}\n"
                    f"Sample Data: {df.head(3).to_string(index=False)}\n\n"
                    f"Question: {user_question}"
                )
                hf_token = st.secrets["hf_token"]
                with st.spinner("Getting answer from AI..."):
                    ai_response = query_huggingface(question_prompt, hf_token)

                cleaned = ai_response.strip()
                lines = [line.strip() for line in cleaned.split("\n") if line.strip() and not line.lower().startswith("as an ai")]
                last_line = lines[-1] if lines else "Response could not be generated."
                st.markdown(
                    f"<div style='background-color:#f0f8f5; padding: 12px; border-radius: 6px; font-size: 15px; white-space: pre-wrap'>{last_line}</div>",
                    unsafe_allow_html=True
                )


        # Check for exact stat questions
        else:
            match = re.match(r".*(mean|average|median|max|min|std).*?(?:of|for)?\s*([a-zA-Z0-9 _%()-]+).*", user_question, re.IGNORECASE)
            if match:
                stat, col_candidate = match.groups()
                stat = stat.lower().strip()
                col_candidate = col_candidate.strip().lower()

                def best_column_match(candidate, df_columns):
                    for col in df_columns:
                        if candidate in col.lower():
                            return col
                    return None

                matched_col = best_column_match(col_candidate, df.columns)

                if matched_col and matched_col in df.select_dtypes(include='number').columns:
                    stat_map = {
                        'mean': 'mean',
                        'average': 'mean',
                        'median': '50%',
                        'max': 'max',
                        'min': 'min',
                        'std': 'std'
                    }
                    stat_key = stat_map.get(stat)
                    if stat_key:
                        value = df[matched_col].describe().get(stat_key)
                        if value is not None:
                            st.markdown(
                                f"<div style='background-color:#f0f8f5; padding: 12px; border-radius: 6px; font-size: 15px;'>The {stat} of '<b>{matched_col}</b>' is <b>{value:.4f}</b>.</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.warning("Could not find the requested statistic.")
                    else:
                        st.warning("Unsupported statistic requested.")
                else:
                    st.warning("Could not match the column for your question.")
            else:
                # Fallback to LLM if it's not a clean stat question
                question_prompt = (
                    f"Answer the following based on the dataset:\n"
                    f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}\n"
                    f"Sample Data: {df.head(3).to_string(index=False)}\n\n"
                    f"Question: {user_question}"
                )
                hf_token = st.secrets["hf_token"]
                with st.spinner("Getting answer from AI..."):
                    ai_response = query_huggingface(question_prompt, hf_token)

                cleaned = ai_response.strip()
                lines = [line.strip() for line in cleaned.split("\n") if line.strip() and not line.lower().startswith("as an ai")]
                last_line = lines[-1] if lines else "Response could not be generated."
                st.markdown(
                    f"<div style='background-color:#f0f8f5; padding: 12px; border-radius: 6px; font-size: 15px; white-space: pre-wrap'>{last_line}</div>",
                    unsafe_allow_html=True
                )
