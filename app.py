
import os
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

from tools import (
    dataset_overview,
    filter_data,
    calculate_statistics,
    group_analysis,
    correlation_analysis,
)

load_dotenv()

st.set_page_config(
    page_title="CSV Analysis Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 CSV Analysis Agent")
st.write(
    "Upload a CSV file, ask a question, and let Gemini "
    "choose the appropriate data-analysis tool."
)


def make_json_safe(result):
    """Convert tool results into values Gemini can understand."""
    if isinstance(result, pd.DataFrame):
        return {
            "columns": [str(c) for c in result.columns],
            "rows": json.loads(
                result.to_json(orient="records", default_handler=str)
            ),
        }

    if isinstance(result, pd.Series):
        return result.to_dict()

    if hasattr(result, "item"):
        return result.item()

    if isinstance(result, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in result.items()
        }

    if isinstance(result, (list, tuple)):
        return [make_json_safe(v) for v in result]

    return result


uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("CSV uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))

        st.write("**Column names:**", ", ".join(map(str, df.columns)))

        # Functions available for Gemini to call.
        # Each function uses the uploaded DataFrame.

        def overview_tool() -> dict:
            """Show dataset row count, column count, column types,
            and missing values."""
            return make_json_safe(dataset_overview(df))

        def statistics_tool(column: str) -> dict:
            """Calculate count, mean, median, minimum and maximum
            for a numeric column. The column must be an exact
            column name from the uploaded dataset."""
            if column not in df.columns:
                raise ValueError(f"Unknown column: {column}")
            return make_json_safe(calculate_statistics(df, column))

        def filter_tool(
            column: str,
            operator: str,
            value: str,
        ) -> dict:
            """Filter dataset rows. Supported operators are equals,
            contains, >, <, >= and <=. Use an exact column name.
            The value is the value to compare against."""
            if column not in df.columns:
                raise ValueError(f"Unknown column: {column}")

            result = filter_data(df, column, operator, value)
            return make_json_safe(result)

        def group_tool(
            category: str,
            value: str,
            operation: str = "mean",
        ) -> dict:
            """Group data by a category column and calculate mean,
            sum or count for a value column. Use exact column names."""
            if category not in df.columns:
                raise ValueError(f"Unknown category column: {category}")
            if value not in df.columns:
                raise ValueError(f"Unknown value column: {value}")
            if operation not in ["mean", "sum", "count"]:
                raise ValueError("Operation must be mean, sum or count")

            result = group_analysis(df, category, value, operation)
            return make_json_safe(result)

        def correlation_tool() -> dict:
            """Calculate the correlation matrix for numeric columns."""
            return make_json_safe(correlation_analysis(df))

        st.subheader("Ask a question about your data")

        question = st.text_input(
            "Your question",
            placeholder="Which subject has the highest average marks?",
        )

        if st.button("Ask AI", type="primary"):
            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                st.error(
                    "Gemini API key not found. Add GEMINI_API_KEY "
                    "to your .env file and restart the app."
                )

            elif not question.strip():
                st.warning("Please enter a question first.")

            else:
                try:
                    client = genai.Client(api_key=api_key)

                    with st.spinner("Gemini is analysing your data..."):
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=(
                                "You are a CSV analysis agent. Answer "
                                "questions using the supplied tools. "
                                "Use actual tool results for calculations; "
                                "never invent statistics. Use exact dataset "
                                "column names. Explain results simply.\n\n"
                                f"Dataset columns: {list(df.columns)}\n"
                                f"User question: {question}"
                            ),
                            config={
                                "tools": [
                                    overview_tool,
                                    statistics_tool,
                                    filter_tool,
                                    group_tool,
                                    correlation_tool,
                                ],
                            },
                        )

                    if response.text:
                        st.subheader("AI Answer")
                        st.markdown(response.text)
                    else:
                        st.warning(
                            "Gemini returned no text. Try rephrasing "
                            "your question."
                        )

                except Exception as error:
                    st.error(f"Gemini request failed: {error}")

        # Manual tool testing is also available.
        with st.expander("Explore analysis tools manually"):
            choice = st.selectbox(
                "Choose an analysis tool",
                [
                    "Dataset Overview",
                    "Calculate Statistics",
                    "Filter Data",
                    "Group Analysis",
                    "Correlation Analysis",
                ],
            )

            if choice == "Dataset Overview":
                if st.button("Show Dataset Overview"):
                    st.json(make_json_safe(dataset_overview(df)))

            elif choice == "Calculate Statistics":
                numeric_columns = list(
                    df.select_dtypes(include="number").columns
                )
                if numeric_columns:
                    column = st.selectbox(
                        "Numeric column",
                        numeric_columns,
                        key="stats_column",
                    )
                    if st.button("Calculate Statistics"):
                        st.json(
                            make_json_safe(
                                calculate_statistics(df, column)
                            )
                        )
                else:
                    st.info("No numeric columns found.")

            elif choice == "Filter Data":
                column = st.selectbox(
                    "Column to filter",
                    list(df.columns),
                    key="filter_column",
                )
                operator = st.selectbox(
                    "Condition",
                    ["equals", "contains", ">", "<", ">=", "<="],
                )
                value = st.text_input(
                    "Comparison value",
                    key="filter_value",
                )

                if st.button("Filter Rows"):
                    try:
                        result = filter_data(
                            df, column, operator, value
                        )
                        st.dataframe(
                            result, use_container_width=True
                        )
                    except Exception as error:
                        st.error(str(error))

            elif choice == "Group Analysis":
                category = st.selectbox(
                    "Group by",
                    list(df.columns),
                    key="group_category",
                )
                value_column = st.selectbox(
                    "Value column",
                    list(df.columns),
                    key="group_value",
                )
                operation = st.selectbox(
                    "Operation",
                    ["mean", "sum", "count"],
                )

                if st.button("Run Group Analysis"):
                    try:
                        result = group_analysis(
                            df, category, value_column, operation
                        )
                        st.write(result)
                    except Exception as error:
                        st.error(str(error))

            elif choice == "Correlation Analysis":
                if st.button("Calculate Correlations"):
                    try:
                        st.dataframe(
                            correlation_analysis(df),
                            use_container_width=True,
                        )
                    except Exception as error:
                        st.error(str(error))

    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")

else:
    st.info("Upload a CSV file to get started.")
