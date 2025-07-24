import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import io
import plotly.io as pio
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def main():
    """Main function to run the Streamlit Data Explorer app."""
    st.sidebar.header("App Settings")
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
    if dark_mode:
        st.write('<style>body { background-color: #222; color: #eee; }</style>', unsafe_allow_html=True)
        st.set_page_config(page_title="Data Explorer", layout="wide", initial_sidebar_state="expanded")
    else:
        st.set_page_config(page_title="Data Explorer", layout="wide", initial_sidebar_state="expanded")
    st.title("📊 Data Explorer – Visualize and Analyze Any CSV File")
    st.sidebar.header("1. Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        handle_file_upload(uploaded_file)
    else:
        st.info("Please upload a CSV file to get started.")

def handle_file_upload(uploaded_file) -> None:
    """Handle file upload, error checking, and main app logic."""
    try:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File is too large. Please upload a file smaller than 10 MB.")
            return
        try:
            df = load_csv(uploaded_file)
        except UnicodeDecodeError:
            st.error("Encoding error: Please upload a UTF-8 encoded CSV file.")
            return
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        run_data_explorer(df)
    except Exception as e:
        st.error(f"Unexpected error: {e}")

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(uploaded_file)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the DataFrame based on sidebar user input."""
    st.sidebar.markdown("### Data Filter/Search")
    filter_col = st.sidebar.selectbox("Filter by column", df.columns, index=0)
    filter_val = st.sidebar.text_input("Search/filter value")
    if filter_val:
        return df[df[filter_col].astype(str).str.contains(filter_val, case=False, na=False)]
    return df

@st.cache_data(show_spinner=False)
def get_summary_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return describe, missing values, unique values, and mode for the DataFrame."""
    return df.describe(), df.isnull().sum(), df.nunique(), df.mode().iloc[0]

def show_dataset_info(df: pd.DataFrame) -> None:
    """Display dataset shape and column types in the sidebar."""
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.write(f"Shape: {df.shape}")
    st.sidebar.write("Column Types:")
    st.sidebar.write(df.dtypes)

def show_data_preview(df: pd.DataFrame) -> None:
    """Show data preview with show all and pagination options."""
    show_all = st.checkbox("Show all data", value=False)
    if show_all or len(df) <= 10:
        st.dataframe(df)
    else:
        rows_per_page = st.number_input("Rows per page", min_value=5, max_value=100, value=10, step=5)
        total_pages = (len(df) - 1) // rows_per_page + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page - 1) * rows_per_page
        end = start + rows_per_page
        st.dataframe(df.iloc[start:end])

def show_summary_stats(df: pd.DataFrame) -> None:
    """Show summary statistics, missing, unique, and mode values."""
    desc, missing, unique, mode = get_summary_stats(df)
    st.write(desc)
    st.markdown("**Missing Values (per column):**")
    st.write(missing)
    st.markdown("**Unique Values (per column):**")
    st.write(unique)
    st.markdown("**Mode (per column):**")
    st.write(mode)

def clean_data_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar UI for basic data cleaning: drop/fill NAs, convert types, remove duplicates."""
    st.sidebar.markdown("### Data Cleaning Tools")
    clean_df = df.copy()
    # Drop NA
    if st.sidebar.checkbox("Drop rows with missing values"):
        clean_df = clean_df.dropna()
    # Fill NA
    fill_na = st.sidebar.selectbox("Fill missing values with", ["None", "Mean (numeric)", "Median (numeric)", "Zero (numeric)", "Empty string (object)"])
    if fill_na != "None":
        for col in clean_df.columns:
            if clean_df[col].isnull().any():
                if fill_na == "Mean (numeric)" and pd.api.types.is_numeric_dtype(clean_df[col]):
                    clean_df[col] = clean_df[col].fillna(clean_df[col].mean())
                elif fill_na == "Median (numeric)" and pd.api.types.is_numeric_dtype(clean_df[col]):
                    clean_df[col] = clean_df[col].fillna(clean_df[col].median())
                elif fill_na == "Zero (numeric)" and pd.api.types.is_numeric_dtype(clean_df[col]):
                    clean_df[col] = clean_df[col].fillna(0)
                elif fill_na == "Empty string (object)" and pd.api.types.is_object_dtype(clean_df[col]):
                    clean_df[col] = clean_df[col].fillna("")
    # Convert data types
    if st.sidebar.checkbox("Convert columns to numeric where possible"):
        for col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='ignore')
    # Remove duplicates
    if st.sidebar.checkbox("Remove duplicate rows"):
        clean_df = clean_df.drop_duplicates()
    # Export cleaned/filtered data
    csv = clean_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download cleaned data as CSV", csv, file_name="cleaned_data.csv", mime="text/csv")
    return clean_df

def ml_tool_ui(df: pd.DataFrame) -> None:
    """Sidebar UI for simple regression/classification modeling."""
    st.sidebar.markdown("### Quick ML Model")
    ml_task = st.sidebar.selectbox("ML Task", ["None", "Regression", "Classification"])
    if ml_task == "None":
        return
    target_col = st.sidebar.selectbox("Target column", df.columns)
    feature_cols = st.sidebar.multiselect("Feature columns", [col for col in df.columns if col != target_col], default=[col for col in df.columns if col != target_col][:1])
    if not feature_cols:
        st.sidebar.info("Select at least one feature column.")
        return
    test_size = st.sidebar.slider("Test size (fraction)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    if st.sidebar.button("Run Model"):
        X = df[feature_cols]
        y = df[target_col]
        X = pd.get_dummies(X, drop_first=True)  # handle categoricals
        if ml_task == "Regression":
            if not pd.api.types.is_numeric_dtype(y):
                st.sidebar.error("Target must be numeric for regression.")
                return
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("**Regression Results:**")
            st.write(f"R² score: {r2_score(y_test, y_pred):.3f}")
            st.write(f"MSE: {mean_squared_error(y_test, y_pred):.3f}")
            st.write("**Predictions (first 10):**")
            st.write(pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]}))
        elif ml_task == "Classification":
            if not pd.api.types.is_integer_dtype(y) and not pd.api.types.is_bool_dtype(y):
                y = pd.factorize(y)[0]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("**Classification Results:**")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
            st.write("**Predictions (first 10):**")
            st.write(pd.DataFrame({"Actual": y_test[:10], "Predicted": y_pred[:10]}))

def run_data_explorer(df: pd.DataFrame) -> None:
    """Run the main data explorer UI and logic."""
    cleaned_df = clean_data_ui(df)
    filtered_df = filter_dataframe(cleaned_df)
    show_dataset_info(filtered_df)
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Data Preview", expanded=True):
            show_data_preview(filtered_df)
    with col2:
        with st.expander("Summary Statistics", expanded=True):
            show_summary_stats(filtered_df)
    show_visualization_options(filtered_df)
    ml_tool_ui(filtered_df)

def show_visualization_options(filtered_df: pd.DataFrame) -> None:
    """Show chart selection and plotting options in the sidebar and main area."""
    st.sidebar.header("2. Visualization Options")
    numeric_columns = filtered_df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = filtered_df.select_dtypes(exclude=np.number).columns.tolist()
    chart_options = []
    if len(numeric_columns) >= 1:
        chart_options += ["Histogram", "Correlation Heatmap"]
    if len(numeric_columns) >= 1 and len(categorical_columns) >= 1:
        chart_options += ["Bar Chart"]
    if len(numeric_columns) >= 2:
        chart_options += ["Line Chart", "Scatter Plot"]
    if len(categorical_columns) >= 1:
        chart_options += ["Pie Chart"]
    chart_type = st.sidebar.selectbox(
        "Select chart type",
        chart_options if chart_options else ["No available charts"]
    )
    color_palettes = ["Plotly", "Viridis", "Cividis", "Plasma", "Inferno", "Magma", "Turbo"]
    if chart_type == "Bar Chart":
        x_col = st.sidebar.selectbox("X-axis (categorical)", categorical_columns)
        y_cols = st.sidebar.multiselect("Y-axis (numeric, can select multiple)", numeric_columns, default=numeric_columns[:1])
        palette = st.sidebar.selectbox("Color palette", color_palettes)
        if st.sidebar.button("Generate Bar Chart") and y_cols:
            fig = px.bar(filtered_df, x=x_col, y=y_cols, barmode="group", color_discrete_sequence=px.colors.qualitative.__dict__[palette] if palette != "Plotly" else None)
            st.plotly_chart(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button("Download Chart as PNG", buf.getvalue(), file_name="bar_chart.png", mime="image/png")
    elif chart_type == "Line Chart":
        x_col = st.sidebar.selectbox("X-axis (numeric)", numeric_columns)
        y_cols = st.sidebar.multiselect("Y-axis (numeric, can select multiple)", [col for col in numeric_columns if col != x_col], default=[col for col in numeric_columns if col != x_col][:1])
        palette = st.sidebar.selectbox("Color palette", color_palettes)
        if st.sidebar.button("Generate Line Chart") and y_cols:
            fig = px.line(filtered_df, x=x_col, y=y_cols, color_discrete_sequence=px.colors.qualitative.__dict__[palette] if palette != "Plotly" else None)
            st.plotly_chart(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button("Download Chart as PNG", buf.getvalue(), file_name="line_chart.png", mime="image/png")
    elif chart_type == "Scatter Plot":
        x_col = st.sidebar.selectbox("X-axis", numeric_columns)
        y_col = st.sidebar.selectbox("Y-axis", [col for col in numeric_columns if col != x_col])
        palette = st.sidebar.selectbox("Color palette", color_palettes)
        if st.sidebar.button("Generate Scatter Plot"):
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color_discrete_sequence=px.colors.qualitative.__dict__[palette] if palette != "Plotly" else None)
            st.plotly_chart(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button("Download Chart as PNG", buf.getvalue(), file_name="scatter_plot.png", mime="image/png")
    elif chart_type == "Histogram":
        col = st.sidebar.selectbox("Column", numeric_columns)
        bins = st.sidebar.slider("Bins", min_value=5, max_value=100, value=20)
        if st.sidebar.button("Generate Histogram"):
            fig = px.histogram(filtered_df, x=col, nbins=bins)
            st.plotly_chart(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button("Download Chart as PNG", buf.getvalue(), file_name="histogram.png", mime="image/png")
    elif chart_type == "Pie Chart":
        col = st.sidebar.selectbox("Column", categorical_columns)
        if st.sidebar.button("Generate Pie Chart"):
            fig = px.pie(filtered_df, names=col)
            st.plotly_chart(fig, use_container_width=True)
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button("Download Chart as PNG", buf.getvalue(), file_name="pie_chart.png", mime="image/png")
    elif chart_type == "Correlation Heatmap":
        if st.sidebar.button("Show Correlation Heatmap"):
            corr = filtered_df.corr(numeric_only=True)
            fig, ax = plt.subplots()
            mask_strong = abs(corr) > 0.7
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, mask=~mask_strong, linewidths=2, linecolor='black')
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, mask=mask_strong, cbar=False, alpha=0.3)
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("Download Heatmap as PNG", buf.getvalue(), file_name="correlation_heatmap.png", mime="image/png")

if __name__ == "__main__":
    main()
