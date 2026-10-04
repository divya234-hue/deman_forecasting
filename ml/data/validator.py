import pandas as pd
REQUIRED_COLUMNS = ["date", "product_id", "sales"]

def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors=[]
    missing=[column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing: return False, [f"Missing required column: {column}" for column in missing]
    if df.empty: errors.append("Dataset is empty.")
    if df[REQUIRED_COLUMNS].isnull().any().any(): errors.append("Required columns contain missing values.")
    dates=pd.to_datetime(df["date"], errors="coerce")
    if dates.isnull().any(): errors.append("Invalid or missing date values.")
    sales=pd.to_numeric(df["sales"], errors="coerce")
    if sales.isnull().any(): errors.append("Sales must be numeric.")
    elif (sales < 0).any(): errors.append("Sales cannot contain negative values.")
    if df["product_id"].astype(str).str.strip().eq("").any(): errors.append("Product IDs cannot be blank.")
    if not dates.isnull().any() and df.assign(_date=dates).duplicated(["product_id", "_date"]).any():
        errors.append("Duplicate product/date rows are not allowed.")
    for column in ("price", "discount", "promotion", "inventory"):
        if column in df.columns and pd.to_numeric(df[column], errors="coerce").isnull().any():
            errors.append(f"{column} must contain numeric values.")
    return not errors, errors
