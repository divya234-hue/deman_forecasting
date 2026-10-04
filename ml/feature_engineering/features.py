import pandas as pd
def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df
def create_lag_features(
    df: pd.DataFrame,
    group_column: str = "product_id"
) -> pd.DataFrame:

    df = df.copy()
    df = df.sort_values([group_column, "date"])
    df["lag_1"] = df.groupby(group_column)["sales"].shift(1)
    df["lag_7"] = df.groupby(group_column)["sales"].shift(7)
    df["lag_14"] = df.groupby(group_column)["sales"].shift(14)
    df["lag_30"] = df.groupby(group_column)["sales"].shift(30)
    return df
def create_rolling_features(
    df: pd.DataFrame,
    group_column: str = "product_id"
) -> pd.DataFrame:

    df = df.copy()

    df = df.sort_values([group_column, "date"])

    df["rolling_mean_7"] = (
        df.groupby(group_column)["sales"]
        .transform(lambda x: x.shift(1).rolling(7).mean())
    )

    df["rolling_mean_14"] = (
        df.groupby(group_column)["sales"]
        .transform(lambda x: x.shift(1).rolling(14).mean())
    )

    df["rolling_mean_30"] = (
        df.groupby(group_column)["sales"]
        .transform(lambda x: x.shift(1).rolling(30).mean())
    )

    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    return df