
import pandas as pd


def dataset_overview(df):
    """Tool 1: Inspect dataset structure and missing values."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
    }


def filter_data(df, column, operator, value):
    """Tool 2: Filter rows using a condition."""
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")

    series = df[column]

    if operator == "equals":
        mask = series.astype(str) == str(value)
    elif operator == "contains":
        mask = series.astype(str).str.contains(
            str(value), case=False, na=False, regex=False
        )
    elif operator in (">", "<", ">=", "<="):
        numeric_series = pd.to_numeric(series, errors="coerce")
        number = float(value)

        if operator == ">":
            mask = numeric_series > number
        elif operator == "<":
            mask = numeric_series < number
        elif operator == ">=":
            mask = numeric_series >= number
        else:
            mask = numeric_series <= number
    else:
        raise ValueError("Unsupported filter operator")

    return df.loc[mask].head(100)


def calculate_statistics(df, column):
    """Tool 3: Calculate numerical statistics."""
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column}")

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        raise ValueError(f"{column} has no usable numeric values.")

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def group_analysis(df, category, value, operation="mean"):
    """Tool 4: Aggregate numerical values by category."""
    if category not in df.columns or value not in df.columns:
        raise ValueError("Choose valid category and value columns.")

    data = df[[category, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna(subset=[value])

    if operation not in ("mean", "sum", "count"):
        raise ValueError("Operation must be mean, sum, or count.")

    grouped = data.groupby(category, dropna=False)[value]

    if operation == "mean":
        result = grouped.mean()
    elif operation == "sum":
        result = grouped.sum()
    else:
        result = grouped.count()

    return result.sort_values(ascending=False).head(50)


def correlation_analysis(df):
    """Tool 5: Calculate correlations between numeric columns."""
    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        raise ValueError(
            "At least two numeric columns are needed for correlation."
        )

    return numeric.corr()
