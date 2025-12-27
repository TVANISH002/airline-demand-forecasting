import pandas as pd

def load_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df.sort_values("Month").set_index("Month")
    y = df["Passengers"].astype(float).asfreq("MS")

    if y.isna().any():
        y = y.interpolate(method="time")

    return y
