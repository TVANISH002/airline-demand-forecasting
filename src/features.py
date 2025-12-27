import pandas as pd

def make_features(y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y": y})

    for lag in (1, 2, 3, 6, 12):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_mean_6"] = df["y"].shift(1).rolling(6).mean()
    df["roll_std_6"] = df["y"].shift(1).rolling(6).std()

    df["month"] = df.index.month

    return df.dropna()
