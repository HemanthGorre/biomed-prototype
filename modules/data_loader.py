import pandas as pd
import streamlit as st

def upload_data():
    """Streamlit data upload with Excel sheet selection if needed."""
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV uploaded!")
            return df
        elif file_name.endswith('.xlsx'):
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            sheet = st.sidebar.selectbox("Select Excel sheet", sheet_names)
            df = pd.read_excel(xls, sheet_name=sheet)
            st.sidebar.success(f"Excel sheet '{sheet}' loaded!")
            return df
        else:
            st.sidebar.warning("Unsupported file type.")
            return None
    else:
        return None

def data_overview(df):
    st.subheader("Data Overview")
    st.write("**Shape:**", df.shape)
    st.write("**Column Data Types:**")
    st.write(df.dtypes)
    st.write("**Missing Values Summary:**")
    st.write(df.isnull().sum())
    st.write("**Statistical Summary:**")
    st.write(df.describe(include='all').T)
    st.write("**First 5 Rows:**")
    st.write(df.head())

def select_variables(df):
    st.sidebar.header("Step 2: Select Variables")
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    all_cols = df.columns.tolist()
    y_col = st.sidebar.selectbox("Select Dependent Variable (Y)", options=numeric_cols)
    x_cols = st.sidebar.multiselect("Select Independent Variables (X)", options=[col for col in all_cols if col != y_col])
    return y_col, x_cols
