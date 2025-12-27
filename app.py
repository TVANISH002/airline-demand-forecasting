import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.data import load_series
from src.classical import forecast_ets, forecast_arima
from src.ml import recursive_forecast_xgb

DATA_PATH = "data/airline_passenger_timeseries.csv"

st.set_page_config(page_title="Airline Forecasting", page_icon="✈️", layout="wide")

st.markdown(
    """
<style>
/* overall spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* top header card */
.hero {
  padding: 18px 20px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(40,120,255,.20), rgba(130,90,255,.20));
  border: 1px solid rgba(255,255,255,.12);
}
.hero h1 { margin: 0; font-size: 30px; }
.hero p { margin: 6px 0 0 0; opacity: .9; }

/* small stat cards */
.kpi {
  padding: 14px 14px;
  border-radius: 16px;
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.10);
}
.kpi .label { font-size: 12px; opacity: .8; margin-bottom: 4px; }
.kpi .value { font-size: 22px; font-weight: 700; }

/* nicer buttons */
div.stDownloadButton > button, div.stButton > button {
  border-radius: 14px !important;
  padding: 10px 14px !important;
  border: 1px solid rgba(255,255,255,.18) !important;
}

/* section titles */
.section-title { font-size: 18px; font-weight: 800; margin: 4px 0 6px; }

/* subtle divider */
hr { border: none; border-top: 1px solid rgba(255,255,255,.10); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>✈️ Airline Demand Forecasting Dashboard</h1>
  <p>Compare multiple forecasting approaches, pick the champion model, and view uncertainty bands for planning.</p>
</div>
""",
    unsafe_allow_html=True,
)

y = load_series(DATA_PATH)

with st.sidebar:
    st.header("Controls")
    h = st.slider("Forecast horizon (months)", 6, 36, 12)
    z = st.select_slider("Confidence level", options=[1.64, 1.96, 2.58], value=1.96)
    st.caption("1.64≈90%, 1.96≈95%, 2.58≈99%")
    show_raw = st.toggle("Show raw forecast table", value=False)
    show_notes = st.toggle("Show business notes", value=True)

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
    {"ETS": pred_ets, "ARIMA": pred_arima, "XGBoost_Lag": pred_xgb},
    index=future_index
)

df_forecast["Final_Forecast"] = df_forecast["ETS"]

try:
    sigma = float(np.std(ets.resid))
    df_forecast["ETS_Lower"] = df_forecast["Final_Forecast"] - z * sigma
    df_forecast["ETS_Upper"] = df_forecast["Final_Forecast"] + z * sigma
except Exception:
    df_forecast["ETS_Lower"] = np.nan
    df_forecast["ETS_Upper"] = np.nan

# KPIs
latest = float(y.iloc[-1])
forecast_next = float(df_forecast["Final_Forecast"].iloc[0])
avg_forecast = float(df_forecast["Final_Forecast"].mean())

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="kpi"><div class="label">Last actual (latest month)</div><div class="value">{latest:,.0f}</div></div>""",
                unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="kpi"><div class="label">Next month forecast (ETS)</div><div class="value">{forecast_next:,.0f}</div></div>""",
                unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="kpi"><div class="label">Avg forecast (horizon)</div><div class="value">{avg_forecast:,.0f}</div></div>""",
                unsafe_allow_html=True)
with col4:
    try:
        summary = pd.read_csv("outputs/summary_avg_metrics.csv", index_col=0)
        best = summary.sort_values("RMSE").index[0]
        best_rmse = float(summary.loc[best, "RMSE"])
        st.markdown(
            f"""<div class="kpi"><div class="label">Champion (by RMSE)</div><div class="value">{best.upper()}</div>
            <div style="opacity:.8;font-size:12px;margin-top:2px;">RMSE: {best_rmse:.2f}</div></div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            """<div class="kpi"><div class="label">Champion (by RMSE)</div><div class="value">—</div>
            <div style="opacity:.8;font-size:12px;margin-top:2px;">Run train.py</div></div>""",
            unsafe_allow_html=True,
        )

st.markdown("<hr/>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Overview", "✅ Final Forecast", "📊 Model Quality"])

with tab1:
    st.markdown('<div class="section-title">Historical trend</div>', unsafe_allow_html=True)
    st.line_chart(y)

    st.markdown('<div class="section-title">Forecast comparison</div>', unsafe_allow_html=True)
    st.line_chart(df_forecast[["ETS", "ARIMA", "XGBoost_Lag"]])

    if show_raw:
        st.markdown('<div class="section-title">Forecast table</div>', unsafe_allow_html=True)
        st.dataframe(df_forecast)

with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="section-title">Chosen forecast (Production)</div>', unsafe_allow_html=True)
        st.line_chart(df_forecast[["Final_Forecast"]])

        st.markdown('<div class="section-title">Uncertainty band</div>', unsafe_allow_html=True)
        st.line_chart(df_forecast[["ETS_Lower", "Final_Forecast", "ETS_Upper"]])

    with c2:
        st.markdown('<div class="section-title">Download</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download forecast CSV",
            data=df_forecast.to_csv().encode("utf-8"),
            file_name="future_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if show_notes:
            st.markdown('<div class="section-title">Notes</div>', unsafe_allow_html=True)
            st.markdown(
                """
- Strong yearly seasonality + long-term growth  
- ETS selected as champion from walk-forward testing  
- Confidence interval shows realistic range (not a guarantee)  
"""
            )

with tab3:
    st.markdown('<div class="section-title">Backtest summary (avg metrics)</div>', unsafe_allow_html=True)
    try:
        summary = pd.read_csv("outputs/summary_avg_metrics.csv", index_col=0)
        st.dataframe(summary)

        best = summary.sort_values("RMSE").index[0]
        st.success(f"Best model by avg RMSE: {best}")
    except Exception:
        st.info("Run `python train.py` first to generate backtest metrics.")

    with st.expander("What do these metrics mean?"):
        st.write(
            "RMSE penalizes large errors, MAE is average absolute error, and MAPE is percentage error. Lower is better."
        )
