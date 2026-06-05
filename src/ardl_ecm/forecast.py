"""
Forecasting untuk ARDL-ECM monthly:
    - proyeksi exog via VAR
    - forecast monthly Low 1-step ahead dengan bias-corrected log-inverse
"""
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from src.ardl_ecm.config import VAR_MAXLAG, USE_LOG_TRANSFORM
from src.ardl_ecm.model import to_level

logger = logging.getLogger(__name__)


def forecast_exog_var(exog, horizon=1, var_maxlag=VAR_MAXLAG):
    """
    Proyeksi variabel exog bulanan via VAR(p).

    Returns:
        (exog_future, var_info) di mana exog_future DataFrame berindeks bulan ke
        depan, dan var_info dict {k_ar, fallback_p1, ic, var_maxlag} menjelaskan
        bagaimana lag VAR dipilih.
    """
    var_fit = VAR(exog).fit(maxlags=var_maxlag, ic="aic")
    p = var_fit.k_ar
    fallback_p1 = False
    if p == 0:
        # AIC bisa pilih no lag -> fallback ke 1
        var_fit = VAR(exog).fit(1)
        p = 1
        fallback_p1 = True
    seed = exog.values[-p:]
    fc = var_fit.forecast(seed, steps=horizon)

    last_date = exog.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1), periods=horizon, freq="M"
    )
    exog_future = pd.DataFrame(fc, index=future_idx, columns=exog.columns)
    var_info = {
        "k_ar": int(p),
        "fallback_p1": fallback_p1,
        "ic": "aic",
        "var_maxlag": int(var_maxlag),
    }
    return exog_future, var_info


def forecast_monthly(
    fit, endog, exog_future, horizon=1, log_transform=USE_LOG_TRANSFORM
):
    """
    Forecast monthly Low. Inverse transform log -> Rupiah dengan bias
    correction exp(y_hat + sigma2/2).

    Returns:
        (forecast_df, forecast_detail) di mana forecast_df berkolom
        [Predicted_Low, CI_Lower, CI_Upper] berindeks bulan target, dan
        forecast_detail dict {sigma2, yhat_log, ci_lower_log, ci_upper_log,
        log_transform} menjelaskan alur log -> Rupiah secara transparan.
    """
    n = len(endog)
    pred = fit.get_prediction(start=n, end=n + horizon - 1, exog_oos=exog_future)
    frame = pred.summary_frame(alpha=0.05)

    col_mean = "mean" if "mean" in frame.columns else frame.columns[0]
    col_lo = next((c for c in frame.columns if "lower" in c.lower()), None)
    col_hi = next((c for c in frame.columns if "upper" in c.lower()), None)

    sigma2 = float(np.var(fit.resid)) if log_transform else 0.0

    yhat_log = frame[col_mean].values
    lo_log = frame[col_lo].values if col_lo else None
    hi_log = frame[col_hi].values if col_hi else None

    forecast_df = pd.DataFrame(
        {
            "Predicted_Low": to_level(yhat_log, sigma2, log_transform),
            "CI_Lower": to_level(lo_log, sigma2, log_transform)
            if col_lo
            else np.nan,
            "CI_Upper": to_level(hi_log, sigma2, log_transform)
            if col_hi
            else np.nan,
        },
        index=exog_future.index,
    )
    forecast_detail = {
        "sigma2": sigma2,
        "log_transform": bool(log_transform),
        "yhat_log": float(yhat_log[0]),
        "ci_lower_log": float(lo_log[0]) if lo_log is not None else None,
        "ci_upper_log": float(hi_log[0]) if hi_log is not None else None,
    }
    return forecast_df, forecast_detail
