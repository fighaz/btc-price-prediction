"""
Pipeline orchestrator: menjalankan seluruh flow prediksi ARDL
"""
import logging

from config import DEFAULT_MIN_LAG, DEFAULT_MAX_P, DEFAULT_MAX_Q
from src.data import fetch_btc_data
from src.statistics import run_adf_test
from src.optimization import select_optimal_lags
from src.model import train_ardl_model, forecast_future
from src.evaluation import evaluate_forecast
from src.visualization import create_forecast_chart

logger = logging.getLogger(__name__)


def run_forecast(train_end_date, forecast_days=31, min_lag=DEFAULT_MIN_LAG,
                 progress_callback=None):
    """
    Pipeline lengkap: fetch -> ADF -> optimize -> train -> forecast.

    Parameters:
        train_end_date: string "YYYY-MM-DD"
        forecast_days: jumlah hari prediksi
        min_lag: minimum lag offset
        progress_callback: optional callable(step_name, detail) untuk UI

    Returns:
        dict dengan key: df, adf_results, p_lags, q_lags, model, features,
                         train_info, forecast_df, figure
    """
    result = {}

    # 1. Fetch data
    if progress_callback:
        progress_callback("fetch", "Mengambil data dari Indodax API...")
    df = fetch_btc_data(to_date=train_end_date)
    if df is None:
        raise RuntimeError("Gagal fetch data dari API")
    result["df"] = df

    # 2. ADF test
    if progress_callback:
        progress_callback("adf", "Menjalankan ADF test...")
    result["adf_results"] = run_adf_test(df)

    # 3. Optimasi lag
    if progress_callback:
        progress_callback("optimize", "Mencari optimal lag order (AIC)...")
    p_lags, q_lags, lag_results = select_optimal_lags(
        df, max_p=DEFAULT_MAX_P, max_q=DEFAULT_MAX_Q, min_lag=min_lag
    )
    result["p_lags"] = p_lags
    result["q_lags"] = q_lags
    result["lag_results"] = lag_results

    # 4. Train model
    if progress_callback:
        progress_callback("train", "Training ARDL model...")
    ardl_model, df_features, features, train_info = train_ardl_model(
        df, p_lags, q_lags, min_lag=min_lag
    )
    result["model"] = ardl_model
    result["features"] = features
    result["train_info"] = train_info

    # 5. Forecast
    if progress_callback:
        progress_callback("forecast", f"Recursive forecast {forecast_days} hari...")
    forecast_df = forecast_future(
        ardl_model, df, features,
        n_days=forecast_days,
        p_lags=p_lags, q_lags=q_lags, min_lag=min_lag
    )
    result["forecast_df"] = forecast_df

    # 6. Chart
    if progress_callback:
        progress_callback("chart", "Membuat visualisasi...")
    result["figure"] = create_forecast_chart(df, forecast_df)

    return result


def run_forecast_and_evaluate(train_end_date, forecast_days=31, eval_end_date=None,
                              min_lag=DEFAULT_MIN_LAG, progress_callback=None):
    """
    Pipeline lengkap + evaluasi terhadap data aktual.

    Returns:
        dict (sama seperti run_forecast + comparison, metrics, merged_eval, eval_figure)
    """
    result = run_forecast(
        train_end_date, forecast_days, min_lag, progress_callback
    )

    if eval_end_date is None:
        return result

    # Evaluasi
    if progress_callback:
        progress_callback("evaluate", "Mengevaluasi forecast vs data aktual...")

    comparison, metrics, merged_eval = evaluate_forecast(
        result["forecast_df"], eval_end_date
    )

    result["comparison"] = comparison
    result["metrics"] = metrics
    result["merged_eval"] = merged_eval

    # Update chart dengan evaluasi
    if merged_eval is not None:
        if progress_callback:
            progress_callback("eval_chart", "Membuat chart evaluasi...")
        result["figure"] = create_forecast_chart(
            result["df"], result["forecast_df"], merged_eval
        )

    return result
