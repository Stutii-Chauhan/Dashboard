# -*- coding: utf-8 -*-
"""app

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FYybOUoSFapkMlSw4_UdByNuRojkcNTT
"""

import streamlit as st
import pandas as pd

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

    except Exception as e:
        st.error(f"Error reading file: {e}")