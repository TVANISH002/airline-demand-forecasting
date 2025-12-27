import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def fit_ets(train: pd.Series):
    return ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)

def forecast_ets(model, steps: int) -> np.ndarray:
    return model.forecast(steps).values

def fit_arima(train: pd.Series, order=(2, 1, 2)):
    return ARIMA(train, order=order).fit()

def forecast_arima(model, steps: int) -> np.ndarray:
    return model.forecast(steps=steps).values
