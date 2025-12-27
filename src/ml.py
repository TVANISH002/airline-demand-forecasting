import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from .features import make_features

def fit_xgb(train_series: pd.Series) -> XGBRegressor:
    feat = make_features(train_series)
    X = feat.drop(columns=["y"])
    y = feat["y"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X, y)
    return model

def recursive_forecast_xgb(model: XGBRegressor, history: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    preds = []
    hist = history.copy()

    for i, ts in enumerate(future_index):
        feat = make_features(hist)
        X_last = feat.drop(columns=["y"]).iloc[-1:]
        yhat = float(model.predict(X_last)[0])
        preds.append(yhat)
        hist = pd.concat([hist, pd.Series([yhat], index=[ts])])

    return np.array(preds, dtype=float)
