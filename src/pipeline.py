"""
Pipeline orchestrator: menjalankan flow ARDL-ECM monthly.

Tiga entry-point sesuai mode:
    run_monthly_forecast            — forecast 1 bulan ke depan
    run_monthly_forecast_and_evaluate — forecast + evaluasi vs aktual
    run_monthly_backtest            — walk-forward backtest N bulan
"""
import logging

import numpy as np
import pandas as pd

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
                     train_r2, bt=None):
    """Rakit dict ringkasan model untuk disimpan ke DB.

    Bila objek bounds-test `bt` diberikan, sertakan batas kritis lengkap
    (untuk menjelaskan MENGAPA verdict kointegrasi tertentu muncul). Field ini
    hanya dipakai UI; field DB tetap subset yang lama.
    """
    info = {
        "endog_lag": int(endog_lag),
        "exog_orders": str(exog_orders),
        "bounds_f_stat": float(bounds_F),
        "cointegration": cointegration,
        "lambda_ecm": ecm_info["lambda"] if ecm_info else None,
        "half_life_months": ecm_info.get("half_life_months") if ecm_info else None,
        "train_r2": float(train_r2),
    }
    if bt is not None:
        try:
            crit = bt.crit_vals.copy()
            info["bounds_crit_table"] = crit
            info["bounds_lower_5"] = float(crit.loc[95.0, "lower"])
            info["bounds_upper_5"] = float(crit.loc[95.0, "upper"])
        except Exception as e:
            logger.warning(f"Gagal mengambil crit_vals bounds test: {e}")
            info["bounds_crit_table"] = None
    return info


def run_monthly_forecast(
    train_end,
    target_month=None,
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
    Forecast monthly Low untuk bulan target.

    Asumsi: dijalankan di awal bulan M (tanggal 1). Training menggunakan data
    s/d akhir bulan M-1. Harga Open hari pertama bulan M diambil dari API
    dan diinjeksikan ke exog_future menggantikan proyeksi VAR.

    Parameters
    ----------
    train_end    : str tanggal akhir data training (YYYY-MM-DD).
                   Bila target_month diberikan, parameter ini diabaikan dan
                   dihitung otomatis sebagai akhir bulan M-1.
    target_month : str opsional "YYYY-MM" — bulan yang diprediksi.
                   Bila None, dihitung sebagai bulan setelah train_end.

    Returns dict: monthly_history, forecast (DataFrame 1 baris), ardl_info,
    adf_results, ecm_info, figure.
    """
    def step(name, detail):
        if progress_callback:
            progress_callback(name, detail)

    step("fetch", "Mengambil data harian dari Indodax API...")

    # Tentukan bulan target dan batas training
    if target_month is not None:
        target_month_start = pd.Timestamp(target_month + "-01")
    else:
        train_end_dt = pd.Timestamp(train_end)
        target_month_start = train_end_dt + pd.offsets.MonthBegin(1)

    target_year = target_month_start.year
    target_month_num = target_month_start.month

    # Fetch data hingga awal bulan target untuk mendapatkan Open hari pertama
    daily = fetch_btc_daily(to_date=target_month_start.strftime("%Y-%m-%d"))

    # Ambil Open hari pertama bulan target dari data partial
    monthly_with_target = resample_to_monthly(daily, drop_partial=False)
    target_mask = (
        (monthly_with_target.index.year == target_year)
        & (monthly_with_target.index.month == target_month_num)
    )
    current_open = (
        float(monthly_with_target.loc[target_mask, "Open"].iloc[0])
        if target_mask.any() else None
    )

    # Data training: bulan lengkap s/d M-1 (drop_partial=True default)
    monthly_full = resample_to_monthly(daily)
    monthly = apply_rolling_window(monthly_full, rolling_window_years)

    step("adf", "Menjalankan ADF stationarity test...")
    endog, exog = prepare_variables(monthly, log_transform)
    adf_results = check_stationarity(endog, exog)

    step("lag", "Lag selection (information criterion)...")
    endog_lag, exog_orders, lag_ic_table = select_lag_order(
        endog, exog, max_lag_endog, max_lag_exog, ic, trend
    )

    step("estimate", "Estimasi ARDL...")
    fit = estimate_ardl(endog, exog, endog_lag, exog_orders, trend)
    train_r2 = compute_train_r2(fit, endog)

    # Tabel koefisien ARDL (param, koefisien, t-stat, p-value) untuk transparansi.
    coef_table = pd.DataFrame(
        {
            "param": fit.params.index,
            "coef": fit.params.values,
            "t": fit.tvalues.reindex(fit.params.index).values,
            "p": fit.pvalues.reindex(fit.params.index).values,
        }
    )

    step("bounds", "Bounds test kointegrasi...")
    uecm_fit, bt, cointegrated, status = run_bounds_test(
        endog, exog, endog_lag, exog_orders, trend
    )
    ecm_info = interpret_ecm(uecm_fit, cointegrated)

    step("forecast", "Proyeksi exog (VAR) + forecast bulan ini...")
    exog_future, var_info = forecast_exog_var(exog, horizon=1, var_maxlag=var_maxlag)
    # Simpan proyeksi VAR murni sebelum injeksi Open (untuk perbandingan di UI).
    var_info["exog_var_only"] = exog_future.copy()
    # Injeksi actual Open bulan target bila sudah tersedia
    if current_open is not None:
        col_open = "log_Open" if log_transform else "Open"
        actual_open_val = np.log(current_open) if log_transform else current_open
        exog_future.iloc[0, exog_future.columns.get_loc(col_open)] = actual_open_val
    forecast, forecast_detail = forecast_monthly(
        fit, endog, exog_future, horizon=1, log_transform=log_transform
    )

    step("chart", "Membuat visualisasi...")
    figure = plot_single_forecast(forecast, monthly)

    ardl_info = _build_ardl_info(
        endog_lag, exog_orders, bt.stat, status, ecm_info, train_r2, bt=bt
    )
    return {
        "daily": daily,
        "monthly_full": monthly_full,
        "monthly_history": monthly,
        "current_open": current_open,
        "forecast": forecast,
        "forecast_detail": forecast_detail,
        "exog_future": exog_future,
        "var_info": var_info,
        "coef_table": coef_table,
        "lag_ic_table": lag_ic_table,
        "ardl_info": ardl_info,
        "adf_results": adf_results,
        "ecm_info": ecm_info,
        "figure": figure,
    }


def run_monthly_forecast_and_evaluate(
    train_end,
    eval_end,
    target_month=None,
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
        train_end, target_month=target_month,
        log_transform=log_transform, rolling_window_years=rolling_window_years,
        max_lag_endog=max_lag_endog, max_lag_exog=max_lag_exog,
        ic=ic, trend=trend, var_maxlag=var_maxlag,
        progress_callback=progress_callback,
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

    Konsisten dengan asumsi dijalankan di awal bulan: tiap iterasi hanya
    menggunakan data s/d tanggal 1 bulan target (Open hari pertama diinjeksikan
    ke exog_future, Low/Close aktual dipakai hanya untuk evaluasi).

    train_end diterima sebagai akhir bulan terakhir yang di-backtest, lalu
    diubah ke tanggal 1 bulan itu agar fetch konsisten.

    Returns dict: daily, monthly_history, results_df, metrics, figure.
    """
    if progress_callback:
        progress_callback("fetch", "Mengambil data harian dari Indodax API...")

    # Konsistensi: fetch s/d tanggal 1 bulan terakhir backtest
    # (bukan akhir bulan) — sama dengan asumsi Forecast.
    # resample drop_partial=False memastikan bulan terakhir masuk sebagai actual_row.
    train_end_dt = pd.Timestamp(train_end)
    last_month_start = train_end_dt.replace(day=1)
    daily = fetch_btc_daily(to_date=last_month_start.strftime("%Y-%m-%d"))
    # drop_partial=True: bulan berjalan (belum tutup) tidak masuk sebagai
    # actual_row backtest — actual_low harus dari bulan yang sudah lengkap.
    monthly = resample_to_monthly(daily, drop_partial=True)

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
        "daily": daily,
        "monthly_history": monthly,
        "results_df": results_df,
        "metrics": metrics,
        "figure": figure,
    }
