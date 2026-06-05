"""
ARDL-ECM core: transform, stationarity, lag selection, estimation,
bounds test, dan ECM diagnostics.

Semua fungsi menerima parameter eksplisit (log_transform, max_lag, ic, trend)
dengan default dari src.ardl_ecm.config — supaya bisa di-override per-run.
"""
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch, breaks_cusumolsresid
from statsmodels.stats.stattools import jarque_bera

from src.ardl_ecm.config import (
    MAX_LAG_ENDOG,
    MAX_LAG_EXOG,
    IC,
    TREND,
    USE_LOG_TRANSFORM,
    ROLLING_WINDOW_YEARS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS: log/level transform
# ============================================================================
def to_log(series, log_transform=USE_LOG_TRANSFORM):
    return np.log(series) if log_transform else series


def to_level(series, sigma2=0.0, log_transform=USE_LOG_TRANSFORM):
    """Inverse transform log -> level dengan bias correction exp(y_hat + sigma2/2)."""
    if log_transform:
        return np.exp(series + sigma2 / 2)
    return series


def apply_rolling_window(monthly, rolling_window_years=ROLLING_WINDOW_YEARS):
    """Potong data ke N tahun terakhir; 0 = semua data."""
    if rolling_window_years > 0:
        window = rolling_window_years * 12
        return monthly.iloc[-window:] if len(monthly) > window else monthly
    return monthly


# ============================================================================
# PREPARE VARIABLES
# ============================================================================
def prepare_variables(monthly, log_transform=USE_LOG_TRANSFORM):
    """Pisahkan endog (Low) dan exog (Open, Close, log_Volume)."""
    if log_transform:
        endog = np.log(monthly["Low"]).copy()
        exog = pd.DataFrame(
            {
                "log_Open": np.log(monthly["Open"]),
                "log_Close": np.log(monthly["Close"]),
                "log_Volume": monthly["log_Volume"],
            },
            index=monthly.index,
        )
    else:
        endog = monthly["Low"].copy()
        exog = monthly[["Open", "Close", "log_Volume"]].copy()
    return endog, exog


# ============================================================================
# ADF STATIONARITY
# ============================================================================
def check_stationarity(endog, exog):
    """ADF test level + first-difference. Returns dict per variabel.

    Tiap entri memuat statistik + nilai kritis 1/5/10% + nobs + lag yang dipakai,
    agar transparan MENGAPA sebuah variabel disebut stasioner/tidak (bandingkan
    stat vs nilai kritis, bukan hanya p-value).
    """
    series_map = {"Low": endog, **{c: exog[c] for c in exog.columns}}
    results = {}
    for name, s in series_map.items():
        adf_stat, p_val, used_lag, nobs, crit, _ = adfuller(s.dropna(), autolag="AIC")
        stationary = p_val < 0.05
        results[name] = {
            "adf_stat": float(adf_stat),
            "p_value": float(p_val),
            "level_stationary": stationary,
            "used_lag": int(used_lag),
            "nobs": int(nobs),
            "crit_1": float(crit["1%"]),
            "crit_5": float(crit["5%"]),
            "crit_10": float(crit["10%"]),
        }

    non_stat = [n for n, r in results.items() if not r["level_stationary"]]
    if non_stat:
        for name in non_stat:
            diff = series_map[name].diff().dropna()
            adf_stat, p_val, used_lag, nobs, crit, _ = adfuller(diff, autolag="AIC")
            results[name]["diff_adf_stat"] = float(adf_stat)
            results[name]["diff_p_value"] = float(p_val)
            results[name]["diff_used_lag"] = int(used_lag)
            results[name]["diff_nobs"] = int(nobs)
            results[name]["diff_crit_1"] = float(crit["1%"])
            results[name]["diff_crit_5"] = float(crit["5%"])
            results[name]["diff_crit_10"] = float(crit["10%"])
            results[name]["order"] = 1 if p_val < 0.05 else 2
    else:
        for name in results:
            results[name]["order"] = 0

    results["_any_i2"] = any(
        r.get("order") == 2 for k, r in results.items() if isinstance(r, dict)
    )
    return results


# ============================================================================
# LAG SELECTION
# ============================================================================
def select_lag_order(
    endog, exog, max_lag_endog=MAX_LAG_ENDOG, max_lag_exog=MAX_LAG_EXOG,
    ic=IC, trend=TREND,
):
    """Lag selection via information criterion.

    Grid search semua kombinasi (lag endogen × order tiap eksogen), pilih IC
    terkecil. Returns (endog_lag, exog_orders, ic_table) di mana ic_table adalah
    DataFrame seluruh kandidat [Terpilih, Lag (endog, exog), IC] terurut menaik —
    untuk membuktikan MENGAPA kombinasi lag tertentu terpilih. ic_table bisa None
    bila struktur sel.aic/sel.bic tidak terbaca.
    """
    sel = ardl_select_order(
        endog=endog,
        maxlag=max_lag_endog,
        exog=exog,
        maxorder=max_lag_exog,
        ic=ic,
        trend=trend,
    )
    exog_cols = list(exog.columns)

    def _norm_orders(orders):
        """Normalkan dict order: pakai semua kolom exog, None -> 0 (lag 0)."""
        return {c: (0 if orders.get(c) is None else int(orders.get(c)))
                for c in exog_cols}

    def _norm_lag(p):
        """Normalkan lag endogen: None -> 0."""
        return 0 if p is None else int(p)

    # Sumber kebenaran: baris pertama sel.aic/sel.bic — value (endog_lag,
    # exog_orders_dict) per-nama. Lebih andal daripada sel.model.ardl_order
    # (yang bisa berbeda panjang dengan jumlah kolom). Tabel IC & nilai terpilih
    # diturunkan dari SUMBER YANG SAMA agar konsisten.
    ic_series = sel.aic if ic == "aic" else sel.bic

    endog_lag = None
    exog_orders = None
    ic_table = None
    try:
        items = list(ic_series.items())
        # Baris pertama (IC terkecil) = kombinasi terpilih.
        best_p, best_orders = items[0][1]
        endog_lag = _norm_lag(best_p)
        exog_orders = _norm_orders(best_orders)

        rows = []
        for ic_val, (p, orders) in items:
            rows.append({
                "Lag (endog, exog)": f"({_norm_lag(p)}, {_norm_orders(orders)})",
                ic.upper(): float(ic_val),
            })
        ic_table = pd.DataFrame(rows)
        ic_table.insert(0, "Terpilih", "")
        if len(ic_table) > 0:
            ic_table.loc[0, "Terpilih"] = "✓"
    except Exception as e:
        logger.warning(f"Gagal membangun tabel IC lag: {e}")
        ic_table = None

    # Fallback bila parsing sel.aic/sel.bic gagal: pakai ardl_order (per posisi).
    if endog_lag is None or exog_orders is None:
        ardl_order = sel.model.ardl_order
        endog_lag = int(ardl_order[0])
        exog_orders = _norm_orders(
            dict(zip(exog_cols, ardl_order[1:]))
        )

    return endog_lag, exog_orders, ic_table


# ============================================================================
# ARDL ESTIMATION
# ============================================================================
def estimate_ardl(endog, exog, endog_lag, exog_orders, trend=TREND):
    """Estimasi ARDL via OLS. Returns statsmodels fit object."""
    model = ARDL(
        endog=endog, lags=endog_lag, exog=exog, order=exog_orders, trend=trend
    )
    return model.fit()


def compute_train_r2(fit, endog):
    """In-sample R2 dari fitted values."""
    fitted = fit.fittedvalues
    y_true = endog.loc[fitted.index]
    ss_res = np.sum((y_true - fitted) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ============================================================================
# BOUNDS TEST
# ============================================================================
def run_bounds_test(endog, exog, endog_lag, exog_orders, trend=TREND):
    """
    Bounds test Pesaran-Shin-Smith untuk kointegrasi.

    Returns:
        (uecm_fit, bounds_test, cointegrated_bool, status_string)
    """
    # UECM requires endog lag >= 1 dan semua exog lag >= 1
    uecm_lag = max(endog_lag, 1)
    uecm_orders = {k: max(v, 1) for k, v in exog_orders.items()}
    uecm = UECM(
        endog=endog, lags=uecm_lag, exog=exog, order=uecm_orders, trend=trend
    ).fit()
    bt = uecm.bounds_test(case=3)

    f_stat = bt.stat
    try:
        upper_5 = bt.crit_vals.loc[0.05, "upper"]
        lower_5 = bt.crit_vals.loc[0.05, "lower"]
    except KeyError:
        idx = bt.crit_vals.index
        idx_5 = idx[np.argmin(np.abs(np.array(idx) - 0.05))]
        upper_5 = bt.crit_vals.loc[idx_5, "upper"]
        lower_5 = bt.crit_vals.loc[idx_5, "lower"]

    if f_stat > upper_5:
        status, cointegrated = "COINTEGRATED", True
    elif f_stat < lower_5:
        status, cointegrated = "NOT_COINTEGRATED", False
    else:
        status, cointegrated = "INCONCLUSIVE", True

    return uecm, bt, cointegrated, status


# ============================================================================
# ECM INTERPRETATION
# ============================================================================
def interpret_ecm(uecm_fit, cointegrated):
    """
    Diagnostics UECM: speed-of-adjustment lambda, half-life, long-run coefs,
    residual diagnostics. Returns dict atau None kalau tidak kointegrasi.
    """
    if not cointegrated:
        return None

    params = uecm_fit.params
    tvalues = uecm_fit.tvalues
    pvalues = uecm_fit.pvalues

    lambda_name = None
    for candidate in ["Low.L1", "Low.L1.level", "y.L1"]:
        if candidate in params.index:
            lambda_name = candidate
            break
    if lambda_name is None:
        for name in params.index:
            if "L1" in name and name.startswith(("Low", "y")):
                lambda_name = name
                break
    if lambda_name is None:
        return None

    lam = params[lambda_name]
    t_lam = tvalues[lambda_name]
    p_lam = pvalues[lambda_name]
    half_life = np.log(0.5) / np.log(1 + lam) if -1 < lam < 0 else np.nan

    # Long-run coefficients
    lr_coefs = {}
    level_params = {
        k: v
        for k, v in params.items()
        if not k.startswith("D.") and k != lambda_name and k != "const"
    }
    for name, coef in level_params.items():
        lr = -coef / lam if lam != 0 else np.nan
        var_label = name.split(".")[0]
        lr_coefs[var_label] = {
            "uecm_coef": float(coef),
            "lr_coef": float(lr),
            "t": float(tvalues.get(name, np.nan)),
            "p": float(pvalues.get(name, np.nan)),
        }

    result = {
        "lambda": float(lam),
        "t": float(t_lam),
        "p": float(p_lam),
        "half_life_months": float(half_life) if np.isfinite(half_life) else None,
        "long_run": lr_coefs,
    }

    resid = uecm_fit.resid

    # Breusch-Godfrey serial correlation (lag=4) via manual LM
    try:
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm
        from scipy import stats as scipy_stats

        n_resid = len(resid)
        nlags = 4
        resid_arr = resid.values if hasattr(resid, "values") else np.array(resid)
        lag_mat = np.column_stack(
            [resid_arr[nlags - j : n_resid - j] for j in range(1, nlags + 1)]
        )
        y_bg = resid_arr[nlags:]
        X_bg = sm.add_constant(lag_mat)
        bg_fit = OLS(y_bg, X_bg).fit()
        bg_lm = len(y_bg) * bg_fit.rsquared
        result["bg_lm"] = float(bg_lm)
        result["bg_p"] = float(scipy_stats.chi2.sf(bg_lm, df=nlags))
    except Exception as e:
        logger.warning(f"BG test gagal: {e}")
        result["bg_p"] = None

    # ARCH LM heteroskedasticity (lag=4)
    try:
        arch_lm, arch_p, _, _ = het_arch(resid, nlags=4)
        result["arch_lm"] = float(arch_lm)
        result["arch_p"] = float(arch_p)
    except Exception:
        result["arch_p"] = None

    # Jarque-Bera normality
    try:
        jb_stat, jb_p, skew, kurt = jarque_bera(resid)
        result["jb_stat"] = float(jb_stat)
        result["jb_p"] = float(jb_p)
        result["skew"] = float(skew)
        result["kurt"] = float(kurt)
    except Exception:
        result["jb_p"] = None

    # CUSUM structural stability
    try:
        cusum_result = breaks_cusumolsresid(resid)
        cusum_arr = cusum_result[0]
        cusum_crit = cusum_result[1]
        result["cusum_stable"] = bool(np.all(np.abs(cusum_arr) <= cusum_crit))
    except Exception as e:
        logger.warning(f"CUSUM test gagal: {e}")
        result["cusum_stable"] = None

    return result
