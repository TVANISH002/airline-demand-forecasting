# ✈️ Airline Demand Forecasting

Real-world time series forecasting project to support airline planning and decision-making.

## Why this project
Airlines rely on demand forecasts to plan capacity, staffing, and budgets. Inaccurate forecasts lead to wasted resources or lost revenue. This project demonstrates how historical passenger data can be used to produce reliable, business-ready demand forecasts.

## What I built
- Forecasted future airline passenger demand using historical monthly data  
- Compared baseline, statistical, and machine-learning forecasting approaches  
- Selected the most reliable model using realistic time-series evaluation  
- Delivered results through an interactive Streamlit dashboard  

## How it works
- Used naive and seasonal naive forecasts as benchmarks  
- Applied ETS (Holt-Winters) and ARIMA for statistical forecasting  
- Built an XGBoost model using lag-based features  
- Evaluated models with walk-forward validation to avoid data leakage  

## Key outcome
- ETS (Holt-Winters) produced the most stable and accurate forecasts  
- Final forecasts include confidence intervals to support risk-aware planning  
- Results are easy to explore and export via the dashboard  

## Tools
Python, Pandas, NumPy, statsmodels, XGBoost, Streamlit


### Limitations
- The model is trained on historical passenger demand from a single airline route
  or dataset and may not generalize to other airlines, regions, or market conditions.
- External factors such as fuel prices, economic conditions, weather events,
  and regulatory changes are not explicitly modeled.
- Forecasts assume historical seasonality patterns will continue into the future.
- The machine-learning model relies on lag-based features and does not capture
  sudden structural breaks (e.g., pandemics, shocks) without retraining.



