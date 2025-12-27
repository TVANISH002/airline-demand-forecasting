from pathlib import Path
import pandas as pd
import joblib

from src.data import load_series
from src.backtest import walk_forward_backtest
from src.classical import fit_ets, fit_arima
from src.ml import fit_xgb

DATA_PATH = "data/airline_passenger_timeseries.csv"
ARIMA_ORDER = (2, 1, 2)

def main():
    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    y = load_series(DATA_PATH)

    metrics_df, preds_df = walk_forward_backtest(
        y,
        horizon=12,
        folds=5,
        arima_order=ARIMA_ORDER
    )

    metrics_df.to_csv("outputs/backtest_metrics.csv", index=False)
    preds_df.to_csv("outputs/backtest_predictions.csv", index=False)

    avg_rmse = metrics_df.groupby("model")["RMSE"].mean().sort_values()
    champion = avg_rmse.index[0]

    summary = pd.DataFrame({
        "avg_RMSE": avg_rmse,
        "avg_MAPE": metrics_df.groupby("model")["MAPE"].mean(),
        "avg_MAE": metrics_df.groupby("model")["MAE"].mean(),
    }).sort_values("avg_RMSE")

    summary.to_csv("outputs/summary_avg_metrics.csv")
    print("\nAverage metrics (lower is better):")
    print(summary)

    ets_model = fit_ets(y)
    arima_model = fit_arima(y, order=ARIMA_ORDER)
    xgb_model = fit_xgb(y)

    joblib.dump(ets_model, "models/ets_model.joblib")
    joblib.dump(arima_model, "models/arima_model.joblib")
    joblib.dump(xgb_model, "models/xgb_model.joblib")
    joblib.dump({"arima_order": ARIMA_ORDER, "champion_by_rmse": champion}, "models/meta.joblib")

    print(f"\nChampion (by avg RMSE): {champion}")
    print("Saved outputs/ and models/")

if __name__ == "__main__":
    main()
