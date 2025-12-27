import numpy as np
import pandas as pd

from .metrics import mae, rmse, mape
from .classical import fit_ets, forecast_ets, fit_arima, forecast_arima
from .ml import fit_xgb, recursive_forecast_xgb

def naive_forecast(train: pd.Series, steps: int) -> np.ndarray:
    return np.repeat(train.iloc[-1], steps)

def seasonal_naive_forecast(train: pd.Series, steps: int, season_len: int = 12) -> np.ndarray:
    last_season = train.values[-season_len:]
    reps = int(np.ceil(steps / season_len))
    return np.tile(last_season, reps)[:steps]

def walk_forward_backtest(y: pd.Series, horizon: int = 12, folds: int = 5, arima_order=(2, 1, 2)):
    results = []
    preds_rows = []

    total_test = horizon * folds
    start = len(y) - total_test
    if start <= 24:
        raise ValueError("Not enough data for the requested folds/horizon.")

    for f in range(folds):
        fold_start = start + f * horizon
        fold_end = fold_start + horizon

        train = y.iloc[:fold_start]
        test = y.iloc[fold_start:fold_end]

        p_naive = naive_forecast(train, len(test))
        p_snaive = seasonal_naive_forecast(train, len(test))

        ets = fit_ets(train)
        p_ets = forecast_ets(ets, len(test))

        arima = fit_arima(train, order=arima_order)
        p_arima = forecast_arima(arima, len(test))

        xgb = fit_xgb(train)
        p_xgb = recursive_forecast_xgb(xgb, train, test.index)

        model_preds = {
            "naive": p_naive,
            "seasonal_naive": p_snaive,
            "ets": p_ets,
            "arima": p_arima,
            "xgb_lag": p_xgb,
        }

        for name, pred in model_preds.items():
            results.append({
                "fold": f + 1,
                "model": name,
                "MAE": mae(test.values, pred),
                "RMSE": rmse(test.values, pred),
                "MAPE": mape(test.values, pred),
            })

        for i, dt in enumerate(test.index):
            preds_rows.append({
                "fold": f + 1,
                "date": dt,
                "y_true": float(test.iloc[i]),
                "naive": float(p_naive[i]),
                "seasonal_naive": float(p_snaive[i]),
                "ets": float(p_ets[i]),
                "arima": float(p_arima[i]),
                "xgb_lag": float(p_xgb[i]),
            })

    return pd.DataFrame(results), pd.DataFrame(preds_rows)
