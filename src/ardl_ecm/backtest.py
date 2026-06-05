"""
Walk-forward rolling backtest & evaluasi untuk ARDL-ECM monthly.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.ardl_ecm.config import (
    MAX_LAG_ENDOG,
    MAX_LAG_EXOG,
    IC,
    TREND,
    VAR_MAXLAG,
    USE_LOG_TRANSFORM,
    ROLLING_WINDOW_YEARS,
    BACKTEST_MONTHS,
)
from src.ardl_ecm.model import (
    prepare_variables,
    select_lag_order,
    estimate_ardl,
    run_bounds_test,
    apply_rolling_window,
)
from src.ardl_ecm.forecast import forecast_exog_var, forecast_monthly
from src.ardl_ecm.data import resample_to_monthly, fetch_btc_daily

logger = logging.getLogger(__name__)


def rolling_backtest(
    monthly,
    n_backtest_months=BACKTEST_MONTHS,
    log_transform=USE_LOG_TRANSFORM,
    rolling_window_years=ROLLING_WINDOW_YEARS,
    max_lag_endog=MAX_LAG_ENDOG,
    max_lag_exog=MAX_LAG_EXOG,
    ic=IC,
    trend=TREND,
    var_maxlag=VAR_MAXLAG,
    progress_callback=None,
):
    """
    Walk-forward backtest: refit setiap iterasi, prediksi Low bulan target.

    Asumsi: dijalankan di awal bulan target — harga Open hari pertama bulan
    target sudah diketahui dan diinjeksikan ke exog_future menggantikan
    proyeksi VAR, sehingga prediksi lebih akurat dan konsisten dengan
    skenario penggunaan nyata (investor menjalankan model di tanggal 1).

    Returns:
        DataFrame results dengan satu baris per bulan target.
    """
    total = len(monthly)
    if total < n_backtest_months + max_lag_endog + 12:
        raise RuntimeError(
            f"Data tidak cukup untuk backtest {n_backtest_months} bulan."
        )

    results = []
    for i in range(n_backtest_months, 0, -1):
        train_end_idx = total - i
        train = monthly.iloc[:train_end_idx]
        actual_row = monthly.iloc[train_end_idx]
        target_dt = actual_row.name
        iter_no = n_backtest_months - i + 1

        try:
            train = apply_rolling_window(train, rolling_window_years)
            endog_t, exog_t = prepare_variables(train, log_transform)
            endog_lag, exog_orders, _ = select_lag_order(
                endog_t, exog_t, max_lag_endog, max_lag_exog, ic, trend
            )
            fit = estimate_ardl(endog_t, exog_t, endog_lag, exog_orders, trend)
            _, bt, _, status = run_bounds_test(
                endog_t, exog_t, endog_lag, exog_orders, trend
            )
            exog_fut, _ = forecast_exog_var(exog_t, horizon=1, var_maxlag=var_maxlag)
            # Injeksi actual Open bulan target: di awal bulan kita sudah tahu
            # harga pembukaan hari pertama, pakai nilai nyata bukan proyeksi VAR.
            col_open = "log_Open" if log_transform else "Open"
            actual_open_val = (
                np.log(actual_row["Open"]) if log_transform else actual_row["Open"]
            )
            exog_fut.iloc[0, exog_fut.columns.get_loc(col_open)] = actual_open_val
            forecast, _ = forecast_monthly(
                fit, endog_t, exog_fut, horizon=1, log_transform=log_transform
            )

            pred_row = forecast.iloc[0]
            pred_low = pred_row["Predicted_Low"]
            ci_lo = pred_row["CI_Lower"]
            ci_hi = pred_row["CI_Upper"]
            actual_low = actual_row["Low"]
            err_pct = abs(actual_low - pred_low) / actual_low * 100
            in_ci = bool(ci_lo <= actual_low <= ci_hi)

            results.append(
                {
                    "month": target_dt,
                    "actual_low": actual_low,
                    "predicted_low": pred_low,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "error_pct": err_pct,
                    "in_ci": in_ci,
                    "bounds_F": bt.stat,
                    "cointegration": status,
                    "endog_lag": endog_lag,
                    "exog_orders": str(exog_orders),
                }
            )
            if progress_callback:
                progress_callback(
                    iter_no,
                    n_backtest_months,
                    f"{target_dt.strftime('%Y-%m')}: err={err_pct:.2f}% ({status})",
                )
        except Exception as e:
            logger.warning(f"Backtest iterasi {target_dt} gagal: {e}")
            results.append(
                {
                    "month": target_dt,
                    "actual_low": actual_row["Low"],
                    "predicted_low": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "error_pct": np.nan,
                    "in_ci": False,
                    "bounds_F": np.nan,
                    "cointegration": "ERROR",
                    "endog_lag": np.nan,
                    "exog_orders": "",
                }
            )
            if progress_callback:
                progress_callback(
                    iter_no, n_backtest_months,
                    f"{target_dt.strftime('%Y-%m')}: GAGAL ({e})",
                )

    return pd.DataFrame(results)


def evaluate_backtest(results_df):
    """Agregasi metrik dari hasil backtest. Returns dict atau None."""
    valid = results_df.dropna(subset=["predicted_low"])
    if valid.empty:
        return None

    actual = valid["actual_low"].values
    predicted = valid["predicted_low"].values
    return {
        "mape": mean_absolute_percentage_error(actual, predicted) * 100,
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "r2": float(r2_score(actual, predicted)),
        "ci_coverage": valid["in_ci"].sum() / len(valid) * 100,
        "n": int(len(valid)),
        "coint_rate": (valid["cointegration"] == "COINTEGRATED").sum()
        / len(valid)
        * 100,
    }


def evaluate_single(forecast_df, eval_to_date):
    """
    Bandingkan 1 bulan forecast dengan aktual yang di-fetch sampai eval_to_date.

    Returns:
        dict hasil evaluasi atau None kalau data aktual belum tersedia.
    """
    actual_daily = fetch_btc_daily(to_date=eval_to_date)
    actual_monthly = resample_to_monthly(actual_daily, drop_partial=False)

    forecast_month = forecast_df.index[0]
    target_period = actual_monthly.index[
        (actual_monthly.index.year == forecast_month.year)
        & (actual_monthly.index.month == forecast_month.month)
    ]
    if target_period.empty:
        return None

    actual_low = actual_monthly.loc[target_period[0], "Low"]
    pred_row = forecast_df.iloc[0]
    err_rp = pred_row["Predicted_Low"] - actual_low
    err_pct = abs(err_rp) / actual_low * 100
    in_ci = bool(pred_row["CI_Lower"] <= actual_low <= pred_row["CI_Upper"])

    return {
        "target_month": forecast_month,
        "actual_low": float(actual_low),
        "predicted_low": float(pred_row["Predicted_Low"]),
        "ci_lower": float(pred_row["CI_Lower"]),
        "ci_upper": float(pred_row["CI_Upper"]),
        "error_rp": float(err_rp),
        "error_pct": float(err_pct),
        "in_ci": in_ci,
    }
