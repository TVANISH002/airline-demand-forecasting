# ✈️ Airline Demand Forecasting

**Time Series Forecasting | Statistical Models | Machine Learning | Streamlit**

## 📌 Project Overview

This project demonstrates an **end-to-end time-series forecasting workflow** using historical airline passenger data.  
The goal is to **forecast future passenger demand**, compare multiple forecasting approaches, select the most reliable model, and present results through an **interactive Streamlit dashboard**.

The focus of the project is on:
- correct handling of time-series data  
- realistic, time-aware model evaluation  
- clear model selection decisions  
- communication of forecast uncertainty  

## 🎯 Problem Statement

Accurate passenger demand forecasting is essential for:
- capacity planning  
- staffing decisions  
- budgeting and risk management  

Using historical airline passenger data, this project answers:
- Which forecasting model performs best on unseen future data?  
- How confident can we be in the predictions?

## 📊 Dataset Description

The dataset used in this project is a **publicly available airline passenger time-series dataset**, commonly used as a benchmark in forecasting problems.

**Key characteristics:**
- Monthly data from **1949 to 1960**
- Target variable: **number of airline passengers**
- Clear **long-term growth** and **yearly seasonality**

This dataset is well suited for demonstrating and comparing baseline, statistical, and machine learning forecasting approaches.

## 🧠 Modeling Approach

### Baseline Models
- Naive Forecast  
- Seasonal Naive Forecast  

Used as benchmarks to validate whether advanced models provide meaningful improvement.

### Statistical Models
- **ETS (Holt-Winters)**  
  - Explicitly models trend and yearly seasonality  
- **ARIMA**  
  - Classical time-series model used for comparison  

### Machine Learning Model
- **XGBoost**
- Uses **lag-based features** to transform the time series into a supervised learning problem  

## 📈 Evaluation Strategy

- **Walk-forward validation** (time-series equivalent of cross-validation)
- Preserves temporal order and avoids future data leakage
- Mimics real-world forecasting scenarios

**Evaluation metrics:**
- RMSE  
- MAE  
- MAPE  

## ✅ Model Selection

All models were evaluated using the same walk-forward framework.

- **ETS (Holt-Winters)** achieved the **lowest average forecast error**
- Selected as the **final production forecast**
- Other models retained for comparison and monitoring

## 📉 Forecast Uncertainty

Forecasting is inherently uncertain.  
To address this, **confidence intervals** were added to the final forecast:

- Final Forecast → best estimate  
- Lower Bound → pessimistic scenario  
- Upper Bound → optimistic scenario  

This allows stakeholders to plan for both expected and extreme outcomes.

## 🖥️ Deployment

An interactive **Streamlit dashboard** was built to:
- visualize historical passenger trends  
- compare forecasts from multiple models  
- present the selected final forecast  
- display confidence intervals  
- export forecast results as a CSV  

## 🛠️ Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **statsmodels** (ETS, ARIMA)
- **XGBoost**
- **Streamlit**
- **joblib**

## 🎓 Skills Demonstrated

- Time Series Analysis  
- Demand Forecasting  
- Statistical Modeling  
- Machine Learning  
- Feature Engineering  
- Model Evaluation & Validation  
- Forecast Uncertainty Estimation  
- Data Visualization  
- Model Deployment  

## 🔮 Future Scope (MLOps & Productionization)

- Integrate **MLflow** for experiment tracking and model registry  
- Package the final model as an API using **BentoML**  
- Add automated retraining as new data becomes available  
- Monitor forecast error and detect model drift  
- Incorporate external features such as holidays or promotions  
- Support user-uploaded time-series datasets  

### ✅ Final Note

This project emphasizes **correct forecasting methodology, evaluation, and communication**, while remaining flexible for future MLOps and production extensions.
## 🚀 Live App

[🔗 Open Streamlit Dashboard](https://airline-demand-forecasting.streamlit.app/)


