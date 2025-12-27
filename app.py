import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.data import load_series
from src.classical import forecast_ets, forecast_arima
from src.ml import recursive_forecast_xgb

DATA_PATH = "data/airline_passenger_timeseries.csv"

st.set_page_config(page_title="Airline Forecasting", layout="wide")
st.title("✈️ Airline Passenger Forecasting")

y = load_series(DATA_PATH)

st.subheader("Historical Passengers")
st.line_chart(y)

h = st.slider("Forecast horizon (months)", 6, 36, 12)

future_index = pd.date_range(
    start=y.index[-1] + pd.offsets.MonthBegin(1),
    periods=h,
    freq="MS"
)

ets = joblib.load("models/ets_model.joblib")
arima = joblib.load("models/arima_model.joblib")
xgb = joblib.load("models/xgb_model.joblib")

pred_ets = forecast_ets(ets, steps=h)
pred_arima = forecast_arima(arima, steps=h)
pred_xgb = recursive_forecast_xgb(xgb, y, future_index)

df_forecast = pd.DataFrame(
    {
        "ETS": pred_ets,
        "ARIMA": pred_arima,
        "XGBoost_Lag": pred_xgb,
    },
    index=future_index
)

df_forecast["Final_Forecast"] = df_forecast["ETS"]

try:
    sigma = float(np.std(ets.resid))
    z = 1.96
    df_forecast["ETS_Lower"] = df_forecast["Final_Forecast"] - z * sigma
    df_forecast["ETS_Upper"] = df_forecast["Final_Forecast"] + z * sigma
except Exception:
    df_forecast["ETS_Lower"] = np.nan
    df_forecast["ETS_Upper"] = np.nan

st.subheader("Forecast Comparison")
st.line_chart(df_forecast[["ETS", "ARIMA", "XGBoost_Lag"]])

st.subheader("Final Forecast")
st.line_chart(df_forecast[["Final_Forecast"]])

st.subheader("Final Forecast with Confidence Interval")
st.line_chart(df_forecast[["ETS_Lower", "Final_Forecast", "ETS_Upper"]])

st.subheader("Notes")
st.markdown(
    """
- Strong yearly seasonality and upward trend  
- ETS selected based on lowest backtest error  
- Other models kept for comparison
"""
)

st.subheader("Backtest Summary")
try:
    summary = pd.read_csv("outputs/summary_avg_metrics.csv", index_col=0)
    st.dataframe(summary)
    best = summary.sort_values("RMSE").index[0]
    st.success(f"Best model: {best}")
except Exception:
    st.info("Run `python train.py` first.")

st.download_button(
    "Download forecast CSV",
    data=df_forecast.to_csv().encode("utf-8"),
    file_name="future_forecast.csv",
    mime="text/csv",
)
