"""
Engine ARDL-ECM monthly untuk prediksi harga terendah bulanan BTC/IDR.

Ekonometrik proper via statsmodels: ADF, lag selection AIC, bounds test
Pesaran-Shin-Smith, UECM speed-of-adjustment, VAR exog projection.
"""
from src.ardl_ecm.data import fetch_btc_daily, resample_to_monthly
from src.ardl_ecm.model import (
    prepare_variables,
    check_stationarity,
    select_lag_order,
    estimate_ardl,
    compute_train_r2,
    run_bounds_test,
    interpret_ecm,
    apply_rolling_window,
    to_log,
    to_level,
)
from src.ardl_ecm.forecast import forecast_exog_var, forecast_monthly
from src.ardl_ecm.backtest import (
    rolling_backtest,
    evaluate_backtest,
    evaluate_single,
)
from src.ardl_ecm.plot import plot_single_forecast, plot_backtest

__all__ = [
    "fetch_btc_daily",
    "resample_to_monthly",
    "prepare_variables",
    "check_stationarity",
    "select_lag_order",
    "estimate_ardl",
    "compute_train_r2",
    "run_bounds_test",
    "interpret_ecm",
    "apply_rolling_window",
    "to_log",
    "to_level",
    "forecast_exog_var",
    "forecast_monthly",
    "rolling_backtest",
    "evaluate_backtest",
    "evaluate_single",
    "plot_single_forecast",
    "plot_backtest",
]
