"""Data-analysis tools the agent can call.

Each function takes a pandas DataFrame plus a few simple arguments and
returns plain Python data (dicts, lists, numbers, strings), so the result
can be sent straight back to the language model as JSON.
"""

import json

import pandas as pd

# Never send more rows than this back to the model.
MAX_ROWS = 20


# ---------- helpers ----------

def clean(value):
    """Make a value JSON-friendly: numpy numbers -> Python numbers, NaN -> None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):  # numpy scalar -> plain Python number
        value = value.item()
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, int):
        return value
    return str(value)


def rows_to_records(df, limit=MAX_ROWS):
    """First `limit` rows of a DataFrame as a list of {column: value} dicts."""
    return json.loads(
        df.head(limit).to_json(orient="records", date_format="iso", default_handler=str)
    )


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def check_column(df, column):
    """Raise a helpful error if `column` isn't in the dataset."""
    if column not in df.columns:
        raise ValueError(
            f"Unknown column '{column}'. Available columns: {list(df.columns)}"
        )


def numeric_values(df, column):
    """The column converted to numbers; text that isn't a number becomes NaN."""
    check_column(df, column)
    if len(df) == 0:
        raise ValueError("There are no rows to analyse (the filter matched nothing).")

    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        # Allow numbers written like "1,250".
        series = series.astype(str).str.replace(",", "", regex=False).str.strip()
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        raise ValueError(f"Column '{column}' has no numeric values.")
    return values


# ---------- tools ----------

def dataset_overview(df):
    """Tool 1: shape, column types, missing values, example values and sample rows."""
    columns = []
    for name in df.columns:
        col = df[name]
        columns.append({
            "name": str(name),
            "type": str(col.dtype),
            "missing": int(col.isna().sum()),
            "unique_values": int(col.nunique()),
            "examples": [str(v) for v in col.dropna().unique()[:3]],
        })
    return {"rows": len(df), "columns": columns, "sample_rows": rows_to_records(df, 3)}


def filter_rows(df, column, operator, value):
    """Tool 2: return only the rows where `column` matches a condition."""
    check_column(df, column)
    series = df[column]
    text = series.astype(str).str.strip().str.lower()
    wanted = str(value).strip().lower()

    if operator in (">", "<", ">=", "<="):
        if is_number(value):
            left, right = numeric_values(df, column), float(value)
        else:
            # Not a number, so try comparing as dates, e.g. Date > 2026-03-01.
            try:
                right = pd.to_datetime(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"'{value}' is not a number or a date, so it can't be used with '{operator}'."
                )
            left = pd.to_datetime(series, errors="coerce")

        if operator == ">":
            mask = left > right
        elif operator == "<":
            mask = left < right
        elif operator == ">=":
            mask = left >= right
        else:
            mask = left <= right

    elif operator in ("equals", "not_equals"):
        if pd.api.types.is_numeric_dtype(series) and is_number(value):
            mask = series == float(value)
        else:
            mask = text == wanted  # case-insensitive text match
        if operator == "not_equals":
            mask = ~mask

    elif operator == "contains":
        mask = text.str.contains(wanted, regex=False, na=False)

    else:
        raise ValueError(
            "operator must be one of: equals, not_equals, contains, >, <, >=, <="
        )

    return df[mask.fillna(False)]


def column_statistics(df, column):
    """Tool 3: summary statistics for one numeric column."""
    values = numeric_values(df, column).dropna()
    return clean({
        "column": column,
        "count": len(values),
        "ignored_missing_or_non_numeric": len(df) - len(values),
        "sum": values.sum(),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "25%": values.quantile(0.25),
        "75%": values.quantile(0.75),
        "max": values.max(),
    })


def group_and_aggregate(df, group_by, value_column, operation="mean"):
    """Tool 4: aggregate a column for each group, sorted from highest to lowest."""
    check_column(df, group_by)
    check_column(df, value_column)

    if operation == "count":
        result = df.groupby(group_by, dropna=False)[value_column].count()
    elif operation in ("mean", "sum", "min", "max", "median"):
        values = numeric_values(df, value_column)
        result = values.groupby(df[group_by], dropna=False).agg(operation)
    else:
        raise ValueError("operation must be one of: mean, sum, count, min, max, median")

    result = result.sort_values(ascending=False)
    return clean({
        "group_by": group_by,
        "value_column": value_column,
        "operation": operation,
        "total_groups": len(result),
        "groups": [{"group": str(k), "value": v} for k, v in result.head(50).items()],
    })


def correlations(df):
    """Tool 5: correlation between every pair of numeric columns, strongest first."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("Need at least two numeric columns to compute correlations.")

    matrix = numeric.corr()
    columns = list(matrix.columns)
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            r = matrix.loc[a, b]
            if pd.notna(r):
                pairs.append({"column_a": str(a), "column_b": str(b), "correlation": round(float(r), 3)})

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return {
        "pairs": pairs[:20],
        "note": "Correlation ranges from -1 to 1. It does not prove causation.",
    }


def top_rows(df, column, n=5, largest=True):
    """Tool 6: the rows with the largest (or smallest) values in a numeric column."""
    values = numeric_values(df, column)
    n = max(1, min(int(n), MAX_ROWS))

    ranked = df.assign(_sort_key=values)
    if largest:
        ranked = ranked.nlargest(n, "_sort_key", keep="all")  # keep ties
    else:
        ranked = ranked.nsmallest(n, "_sort_key", keep="all")
    ranked = ranked.drop(columns="_sort_key")

    return {
        "sorted_by": column,
        "order": "largest first" if largest else "smallest first",
        "rows_returned": len(ranked),
        "rows": rows_to_records(ranked),
    }


def value_counts(df, column, top=10):
    """Tool 7: how often each value appears in a column (counts and percentages)."""
    check_column(df, column)
    counts = df[column].value_counts(dropna=False)
    top = max(1, min(int(top), 50))
    total = len(df)

    return clean({
        "column": column,
        "unique_values": df[column].nunique(),
        "missing": df[column].isna().sum(),
        "values": [
            {"value": str(k), "count": int(c), "percent": 100 * c / total}
            for k, c in counts.head(top).items()
        ],
    })


def find_outliers(df, column):
    """Tool 8: unusually high or low values, using the 1.5 x IQR rule."""
    values = numeric_values(df, column)
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(values < low) | (values > high)]

    return clean({
        "column": column,
        "q1": q1,
        "q3": q3,
        "lower_bound": low,
        "upper_bound": high,
        "outlier_count": len(outliers),
        "outlier_rows": rows_to_records(outliers),
    })


def calculate_column(df, column_a, operation, column_b):
    """Tool 9: compute column_a (add|subtract|multiply|divide) column_b.

    column_b may be another column name or a plain number, e.g. Marks divide 120.
    """
    a = numeric_values(df, column_a)
    if column_b in df.columns:
        b = numeric_values(df, column_b)
    elif is_number(column_b):
        b = float(column_b)
    else:
        raise ValueError(f"'{column_b}' is neither a column name nor a number.")

    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        result = a / b
    else:
        raise ValueError("operation must be one of: add, subtract, multiply, divide")

    # Division by zero gives infinity; treat it as missing instead.
    return result.replace([float("inf"), float("-inf")], float("nan"))
