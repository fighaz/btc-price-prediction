"""
Evaluasi forecast vs data aktual
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from src.data import fetch_btc_data

logger = logging.getLogger(__name__)


def calculate_metrics(actual, predicted):
    """
    Hitung metrik evaluasi: MAPE, RMSE, MAE, R-squared.

    Returns:
        dict dengan key: mape, rmse, mae, r2
    """
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    r2 = r2_score(actual, predicted)

    return {"mape": mape, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_forecast(forecast_df, eval_to_date):
    """
    Fetch data aktual dan bandingkan dengan forecast.

    Parameters:
        forecast_df: hasil dari forecast_future()
        eval_to_date: string "YYYY-MM-DD"

    Returns:
        (comparison_df, metrics_dict, merged_df) atau (None, None, None) jika gagal
    """
    logger.info(f"Evaluasi forecast vs aktual (sampai {eval_to_date})")

    actual_df = fetch_btc_data(to_date=eval_to_date)
    if actual_df is None:
        logger.error("Gagal fetch data aktual untuk evaluasi")
        return None, None, None

    forecast_start = forecast_df["Time"].min()
    forecast_end = forecast_df["Time"].max()

    actual_period = actual_df[
        (actual_df["Time"] >= forecast_start) &
        (actual_df["Time"] <= forecast_end)
    ].copy()

    if len(actual_period) == 0:
        logger.warning(f"Tidak ada data aktual untuk periode {forecast_start.date()} - {forecast_end.date()}")
        return None, None, None

    logger.info(f"Data aktual tersedia: {len(actual_period)} hari")

    # Join berdasarkan tanggal
    forecast_df = forecast_df.copy()
    forecast_df["Date"] = forecast_df["Time"].dt.date
    actual_period["Date"] = actual_period["Time"].dt.date

    merged = pd.merge(
        forecast_df[["Date", "Predicted_Low", "Confidence_Lower", "Confidence_Upper", "Step"]],
        actual_period[["Date", "Low"]].rename(columns={"Low": "Actual_Low"}),
        on="Date",
        how="inner",
    )

    if len(merged) == 0:
        logger.warning("Tidak ada tanggal yang cocok antara forecast dan data aktual")
        return None, None, None

    actual = merged["Actual_Low"].values
    predicted = merged["Predicted_Low"].values

    metrics = calculate_metrics(actual, predicted)

    # CI coverage
    in_ci = ((merged["Actual_Low"] >= merged["Confidence_Lower"]) &
             (merged["Actual_Low"] <= merged["Confidence_Upper"])).sum()
    metrics["ci_coverage"] = in_ci / len(merged) * 100

    # Tabel perbandingan (format rupiah untuk display)
    comparison = pd.DataFrame({
        "Tanggal": merged["Date"].apply(lambda x: x.strftime("%Y-%m-%d")),
        "Low Aktual": merged["Actual_Low"].apply(lambda x: f"Rp {x:,.0f}"),
        "Prediksi": merged["Predicted_Low"].apply(lambda x: f"Rp {x:,.0f}"),
        "Error (Rp)": (merged["Predicted_Low"] - merged["Actual_Low"]).apply(lambda x: f"Rp {x:,.0f}"),
        "Error (%)": ((merged["Predicted_Low"] - merged["Actual_Low"]).abs() / merged["Actual_Low"] * 100).apply(lambda x: f"{x:.2f}%"),
        "Dalam CI": ((merged["Actual_Low"] >= merged["Confidence_Lower"]) &
                     (merged["Actual_Low"] <= merged["Confidence_Upper"])).apply(lambda x: "Ya" if x else "Tidak"),
    })

    logger.info(f"Evaluasi: MAPE={metrics['mape']:.2f}%, CI Coverage={metrics['ci_coverage']:.1f}%")
    return comparison, metrics, merged
