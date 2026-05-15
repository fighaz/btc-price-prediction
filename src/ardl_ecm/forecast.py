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
        DataFrame exog_future berindeks bulan ke depan.
    """
    var_fit = VAR(exog).fit(maxlags=var_maxlag, ic="aic")
    p = var_fit.k_ar
    if p == 0:
        # AIC bisa pilih no lag -> fallback ke 1
        var_fit = VAR(exog).fit(1)
        p = 1
    seed = exog.values[-p:]
    fc = var_fit.forecast(seed, steps=horizon)

    last_date = exog.index[-1]
    future_idx = pd.date_range(
        start=last_date + pd.offsets.MonthEnd(1), periods=horizon, freq="M"
    )
    return pd.DataFrame(fc, index=future_idx, columns=exog.columns)


def forecast_monthly(
    fit, endog, exog_future, horizon=1, log_transform=USE_LOG_TRANSFORM
):
    """
    Forecast monthly Low. Inverse transform log -> Rupiah dengan bias
    correction exp(y_hat + sigma2/2).

    Returns:
        DataFrame [Predicted_Low, CI_Lower, CI_Upper] berindeks bulan target.
    """
    n = len(endog)
    pred = fit.get_prediction(start=n, end=n + horizon - 1, exog_oos=exog_future)
    frame = pred.summary_frame(alpha=0.05)

    col_mean = "mean" if "mean" in frame.columns else frame.columns[0]
    col_lo = next((c for c in frame.columns if "lower" in c.lower()), None)
    col_hi = next((c for c in frame.columns if "upper" in c.lower()), None)

    sigma2 = float(np.var(fit.resid)) if log_transform else 0.0

    return pd.DataFrame(
        {
            "Predicted_Low": to_level(frame[col_mean].values, sigma2, log_transform),
            "CI_Lower": to_level(frame[col_lo].values, sigma2, log_transform)
            if col_lo
            else np.nan,
            "CI_Upper": to_level(frame[col_hi].values, sigma2, log_transform)
            if col_hi
            else np.nan,
        },
        index=exog_future.index,
    )
