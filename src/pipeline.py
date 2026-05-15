"""
Pipeline orchestrator: menjalankan flow ARDL-ECM monthly.

Tiga entry-point sesuai mode:
    run_monthly_forecast            — forecast 1 bulan ke depan
    run_monthly_forecast_and_evaluate — forecast + evaluasi vs aktual
    run_monthly_backtest            — walk-forward backtest N bulan
"""
import logging

import numpy as np

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
from src.ardl_ecm import (
    fetch_btc_daily,
    resample_to_monthly,
    prepare_variables,
    check_stationarity,
    select_lag_order,
    estimate_ardl,
    compute_train_r2,
    run_bounds_test,
    interpret_ecm,
    apply_rolling_window,
    forecast_exog_var,
    forecast_monthly,
    rolling_backtest,
    evaluate_backtest,
    evaluate_single,
    plot_single_forecast,
    plot_backtest,
)

logger = logging.getLogger(__name__)


def _build_ardl_info(endog_lag, exog_orders, bounds_F, cointegration, ecm_info,
                     train_r2):
    """Rakit dict ringkasan model untuk disimpan ke DB."""
    return {
        "endog_lag": int(endog_lag),
        "exog_orders": str(exog_orders),
        "bounds_f_stat": float(bounds_F),
        "cointegration": cointegration,
        "lambda_ecm": ecm_info["lambda"] if ecm_info else None,
        "half_life_months": ecm_info.get("half_life_months") if ecm_info else None,
        "train_r2": float(train_r2),
    }


def run_monthly_forecast(
    train_end,
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
    Forecast monthly Low 1 bulan ke depan.

    Returns dict: monthly_history, forecast (DataFrame 1 baris), ardl_info,
    adf_results, ecm_info, figure.
    """
    def step(name, detail):
        if progress_callback:
            progress_callback(name, detail)

    step("fetch", "Mengambil data harian dari Indodax API...")
    daily = fetch_btc_daily(to_date=train_end)
    monthly_full = resample_to_monthly(daily)
    monthly = apply_rolling_window(monthly_full, rolling_window_years)

    step("adf", "Menjalankan ADF stationarity test...")
    endog, exog = prepare_variables(monthly, log_transform)
    adf_results = check_stationarity(endog, exog)

    step("lag", "Lag selection (information criterion)...")
    endog_lag, exog_orders = select_lag_order(
        endog, exog, max_lag_endog, max_lag_exog, ic, trend
    )

    step("estimate", "Estimasi ARDL...")
    fit = estimate_ardl(endog, exog, endog_lag, exog_orders, trend)
    train_r2 = compute_train_r2(fit, endog)

    step("bounds", "Bounds test kointegrasi...")
    uecm_fit, bt, cointegrated, status = run_bounds_test(
        endog, exog, endog_lag, exog_orders, trend
    )
    ecm_info = interpret_ecm(uecm_fit, cointegrated)

    step("forecast", "Proyeksi exog (VAR) + forecast 1 bulan...")
    exog_future = forecast_exog_var(exog, horizon=1, var_maxlag=var_maxlag)
    forecast = forecast_monthly(
        fit, endog, exog_future, horizon=1, log_transform=log_transform
    )

    step("chart", "Membuat visualisasi...")
    figure = plot_single_forecast(forecast, monthly)

    ardl_info = _build_ardl_info(
        endog_lag, exog_orders, bt.stat, status, ecm_info, train_r2
    )
    return {
        "monthly_history": monthly,
        "forecast": forecast,
        "ardl_info": ardl_info,
        "adf_results": adf_results,
        "ecm_info": ecm_info,
        "figure": figure,
    }


def run_monthly_forecast_and_evaluate(
    train_end,
    eval_end,
    log_transform=USE_LOG_TRANSFORM,
    rolling_window_years=ROLLING_WINDOW_YEARS,
    max_lag_endog=MAX_LAG_ENDOG,
    max_lag_exog=MAX_LAG_EXOG,
    ic=IC,
    trend=TREND,
    var_maxlag=VAR_MAXLAG,
    progress_callback=None,
):
    """Forecast monthly + evaluasi terhadap aktual yang di-fetch sampai eval_end."""
    result = run_monthly_forecast(
        train_end, log_transform, rolling_window_years, max_lag_endog,
        max_lag_exog, ic, trend, var_maxlag, progress_callback,
    )
    if progress_callback:
        progress_callback("evaluate", "Mengevaluasi forecast vs aktual...")
    result["eval_result"] = evaluate_single(result["forecast"], eval_end)

    if result["eval_result"] is not None:
        result["figure"] = plot_single_forecast(
            result["forecast"], result["monthly_history"], result["eval_result"]
        )
    return result


def run_monthly_backtest(
    train_end,
    backtest_months=BACKTEST_MONTHS,
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
    Walk-forward backtest N bulan.

    Returns dict: monthly_history, results_df, metrics, figure.
    """
    if progress_callback:
        progress_callback("fetch", "Mengambil data harian dari Indodax API...")
    daily = fetch_btc_daily(to_date=train_end)
    monthly = resample_to_monthly(daily)

    def bt_progress(iter_no, total, detail):
        if progress_callback:
            progress_callback("backtest", f"[{iter_no}/{total}] {detail}")

    results_df = rolling_backtest(
        monthly,
        n_backtest_months=backtest_months,
        log_transform=log_transform,
        rolling_window_years=rolling_window_years,
        max_lag_endog=max_lag_endog,
        max_lag_exog=max_lag_exog,
        ic=ic,
        trend=trend,
        var_maxlag=var_maxlag,
        progress_callback=bt_progress,
    )
    metrics = evaluate_backtest(results_df)

    figure = None
    if metrics is not None:
        if progress_callback:
            progress_callback("chart", "Membuat visualisasi backtest...")
        figure = plot_backtest(results_df, metrics)

    return {
        "monthly_history": monthly,
        "results_df": results_df,
        "metrics": metrics,
        "figure": figure,
    }
