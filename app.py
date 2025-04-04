import plotly.express as px
#from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import tempfile
from pandas_profiling import ProfileReport

# Upload + Load Data (already done above)
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"Successfully loaded `{uploaded_file.name}`")

        st.subheader("Data Preview")
        st.dataframe(df.head(50))
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # ------------------------------------
        # Generate EDA Report using Profiling
        # ------------------------------------
        st.subheader("Automated EDA Report")
        with st.spinner("Generating EDA..."):
            profile = ProfileReport(df, minimal=True)
            with tempfile.NamedTemporaryFile(suffix=".html") as f:
                profile.to_file(f.name)
                html(open(f.name, "r", encoding="utf-8").read(), height=600, scrolling=True)

        # ------------------------------------
        # Show Example Chart
        # ------------------------------------
        st.subheader("📈 Example Visualization: Top Categories")

        # Auto-detect a categorical column (for demo)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            top_col = cat_cols[0]
            chart_data = df[top_col].value_counts().reset_index()
            chart_data.columns = [top_col, "Count"]
            fig = px.bar(chart_data, x=top_col, y="Count", title=f"Top Values in '{top_col}'")
            st.plotly_chart(fig)
        else:
            st.info("No categorical columns found for bar chart.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
