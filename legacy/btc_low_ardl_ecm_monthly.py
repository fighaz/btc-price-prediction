"""
BTC/IDR Monthly Low Prediction — ARDL-ECM (Econometric, Monthly Frequency)
==========================================================================
Versi monthly dari btc_low_ardl_ecm.py.
Input data tetap harian, lalu di-resample ke bulanan:
    Open       = first day of month
    Close      = last day of month
    Low (Y)    = min of month  ← target prediksi
    Volume     = sum of month  → log_Volume

ARDL-ECM dijalankan pada series bulanan dengan forecast horizon = 1 step (1 bulan).

Mode:
    MODE = "single"           — train s/d TRAIN_END, forecast 1 bulan ke depan,
                                evaluasi vs aktual EVAL_END
    MODE = "rolling_backtest" — walk-forward 12 bulan terakhir, refit setiap iterasi
    MODE = "forecast_only"    — train semua data, forecast 1 bulan ke depan tanpa eval

Cara jalankan:
    python btc_low_ardl_ecm_monthly.py
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_arch, breaks_cusumolsresid
from statsmodels.stats.stattools import jarque_bera
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# KONFIGURASI
# ============================================================================
MODE = "forecast_only"  # "single" | "rolling_backtest" | "forecast_only"
BACKTEST_MONTHS = 12             # jumlah bulan untuk walk-forward eval
TRAIN_END = "2026-06-01"         # cutoff training (mode single)
EVAL_END = "2026-04-30"          # untuk fetch aktual (mode single)

MAX_LAG_ENDOG = 4                # monthly: lag kecil (max 3 bulan)
MAX_LAG_EXOG = 4
IC = "aic"
TREND = "c"
VAR_MAXLAG = 3                   # untuk projeksi exog bulanan

ROLLING_WINDOW_YEARS = 0         # training window (tahun); 0 = semua data sejak 2014
USE_LOG_TRANSFORM = True       # log-transform Y dan X untuk stabilkan variance

OUTPUT_DIR = "output"

API_URL = "https://indodax.com/tradingview/history_v2"
SYMBOL = "BTCIDR"
TIMEFRAME = "1D"
FROM_TS = 1391191321  # 2014-02-01


# ============================================================================
# 1. DATA FETCH (reuse dari daily)
# ============================================================================
def fetch_btc_daily(to_date=None):
    if to_date is None:
        dt = datetime.now()
    else:
        dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

    to_ts = int(dt.timestamp())
    label = dt.strftime("%Y-%m-%d")

    params = {"from": FROM_TS, "to": to_ts, "symbol": SYMBOL, "tf": TIMEFRAME}
    print(f"      Fetching BTC/IDR daily data (sampai {label})...")
    resp = requests.get(API_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal fetch data: HTTP {resp.status_code}")

    df = pd.DataFrame(resp.json())
    df["Time"] = pd.to_datetime(df["Time"], unit="s").dt.normalize()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    df = df.drop_duplicates(subset="Time", keep="last").sort_values("Time").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].replace(0, np.nan).ffill().interpolate(method="linear")

    df = df.set_index("Time").asfreq("D")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].ffill()

    print(f"      Data harian bersih: {len(df)} records ({df.index.min().date()} — {df.index.max().date()})")
    return df


# ============================================================================
# 2. RESAMPLE DAILY → MONTHLY
# ============================================================================
def resample_to_monthly(df_daily, drop_partial=True, min_days=20):
    """
    Aggregation rules (sesuai user pilih):
        Open  = first()  → harga buka hari pertama bulan
        Close = last()   → harga tutup hari terakhir bulan
        Low   = min()    → monthly low (target Y)
        High  = max()    → untuk reference
        Volume= sum()    → total volume bulan, lalu log_Volume = log1p

    drop_partial=True: bulan terakhir dibuang kalau jumlah hari < min_days.
    """
    monthly = pd.DataFrame({
        "Open":   df_daily["Open"].resample("M").first(),
        "High":   df_daily["High"].resample("M").max(),
        "Low":    df_daily["Low"].resample("M").min(),
        "Close":  df_daily["Close"].resample("M").last(),
        "Volume": df_daily["Volume"].resample("M").sum(),
    })

    if drop_partial:
        days_per_month = df_daily["Low"].resample("M").count()
        monthly = monthly[days_per_month >= min_days]

    monthly["log_Volume"] = np.log1p(monthly["Volume"])
    monthly = monthly.dropna()
    print(f"      Data bulanan: {len(monthly)} bulan ({monthly.index.min().strftime('%Y-%m')} — {monthly.index.max().strftime('%Y-%m')})")
    return monthly


# ============================================================================
# HELPERS: log/level transform
# ============================================================================
def to_log(series):
    return np.log(series) if USE_LOG_TRANSFORM else series

def to_level(series, sigma2=0.0):
    """Inverse transform log → level dengan bias correction exp(ŷ + σ²/2)."""
    if USE_LOG_TRANSFORM:
        return np.exp(series + sigma2 / 2)
    return series

def apply_rolling_window(monthly):
    if ROLLING_WINDOW_YEARS > 0:
        window = ROLLING_WINDOW_YEARS * 12
        return monthly.iloc[-window:] if len(monthly) > window else monthly
    return monthly


# ============================================================================
# 3. PREPARE VARIABLES
# ============================================================================
def prepare_variables(monthly):
    if USE_LOG_TRANSFORM:
        endog = np.log(monthly["Low"]).copy()
        exog = pd.DataFrame({
            "log_Open":   np.log(monthly["Open"]),
            "log_Close":  np.log(monthly["Close"]),
            "log_Volume": monthly["log_Volume"],
        }, index=monthly.index)
    else:
        endog = monthly["Low"].copy()
        exog = monthly[["Open", "Close", "log_Volume"]].copy()
    return endog, exog


# ============================================================================
# 4. ADF STATIONARITY
# ============================================================================
def check_stationarity(endog, exog, verbose=True):
    if verbose:
        print("\n[2/8] ADF Stationarity Test (monthly)")
        print("      " + "-" * 68)
        print(f"      {'Variabel':<15} {'ADF Stat':>12} {'p-value':>12} {'Status':<20}")
        print("      " + "-" * 68)

    series_map = {"Low": endog, **{c: exog[c] for c in exog.columns}}
    results = {}
    for name, s in series_map.items():
        adf_stat, p_val = adfuller(s.dropna(), autolag="AIC")[:2]
        stationary = p_val < 0.05
        status = "Stasioner I(0)" if stationary else "Non-stasioner"
        results[name] = {"level_stationary": stationary}
        if verbose:
            print(f"      {name:<15} {adf_stat:>12.4f} {p_val:>12.4f} {status:<20}")

    non_stat = [n for n, r in results.items() if not r["level_stationary"]]
    if non_stat:
        if verbose:
            print("\n      First-Difference Test:")
        for name in non_stat:
            diff = series_map[name].diff().dropna()
            adf_stat, p_val = adfuller(diff, autolag="AIC")[:2]
            results[name]["order"] = 1 if p_val < 0.05 else 2
            if verbose:
                tag = "I(1) OK" if p_val < 0.05 else "I(2) WARNING"
                print(f"      d({name}): ADF={adf_stat:.4f}, p={p_val:.4f} -> {tag}")
    else:
        for name in results:
            results[name]["order"] = 0

    if verbose:
        any_i2 = any(r.get("order") == 2 for r in results.values())
        if any_i2:
            print("\n      PERINGATAN: Ada variabel I(2). ARDL hanya valid untuk campuran I(0)/I(1).")
        else:
            print("\n      OK: Semua variabel I(0) atau I(1). ARDL valid.")
    return results


# ============================================================================
# 5. LAG SELECTION
# ============================================================================
def select_lag_order(endog, exog, verbose=True):
    if verbose:
        print(f"\n[3/8] Lag Selection (ic={IC}, maxlag={MAX_LAG_ENDOG}, maxorder={MAX_LAG_EXOG})")

    sel = ardl_select_order(
        endog=endog,
        maxlag=MAX_LAG_ENDOG,
        exog=exog,
        maxorder=MAX_LAG_EXOG,
        ic=IC,
        trend=TREND,
    )
    ardl_order = sel.model.ardl_order
    endog_lag = ardl_order[0]
    exog_orders = dict(zip(list(exog.columns), ardl_order[1:]))

    if verbose:
        print(f"      Best p (endog lag Low): {endog_lag}")
        for k, v in exog_orders.items():
            print(f"      Best q ({k}): {v}")
    return endog_lag, exog_orders


# ============================================================================
# 6. ARDL ESTIMATION
# ============================================================================
def estimate_ardl(endog, exog, endog_lag, exog_orders, verbose=True):
    model = ARDL(endog=endog, lags=endog_lag, exog=exog, order=exog_orders, trend=TREND)
    fit = model.fit()
    if verbose:
        print("\n[4/8] ARDL Estimation (OLS via statsmodels)")
        fitted = fit.fittedvalues
        y_true = endog.loc[fitted.index]
        ss_res = np.sum((y_true - fitted) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"      Samples used:  {len(fitted)}")
        print(f"      AIC:  {fit.aic:.2f}")
        print(f"      BIC:  {fit.bic:.2f}")
        print(f"      R²  : {r2:.4f}")
        print("      Top 6 coefficients:")
        for name, val in list(fit.params.items())[:6]:
            print(f"        {name:<30} {val:>16,.4f}")
    return fit


# ============================================================================
# 7. BOUNDS TEST
# ============================================================================
def run_bounds_test(endog, exog, endog_lag, exog_orders, verbose=True):
    # UECM requires endog lag >= 1 and all exog lag >= 1 — floor only here, not in ARDL
    uecm_lag = max(endog_lag, 1)
    uecm_orders = {k: max(v, 1) for k, v in exog_orders.items()}
    uecm = UECM(endog=endog, lags=uecm_lag, exog=exog, order=uecm_orders, trend=TREND).fit()
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
        status = "COINTEGRATED"
        cointegrated = True
    elif f_stat < lower_5:
        status = "NOT_COINTEGRATED"
        cointegrated = False
    else:
        status = "INCONCLUSIVE"
        cointegrated = True

    if verbose:
        print("\n[5/8] Bounds Test untuk Kointegrasi (Pesaran-Shin-Smith)")
        print(f"      F-statistic: {f_stat:.4f}")
        print(f"      p-value    : {bt.p_values['upper']:.4f}  (upper-bound)")
        print("      Critical Values:")
        print(f"      {'Level':<8} {'I(0) Lower':>14} {'I(1) Upper':>14}")
        for level in bt.crit_vals.index:
            lo = bt.crit_vals.loc[level, "lower"]
            up = bt.crit_vals.loc[level, "upper"]
            print(f"      {str(level):<8} {lo:>14.4f} {up:>14.4f}")
        print(f"\n      Keputusan @5%: F={f_stat:.4f} vs I(0)={lower_5:.4f}, I(1)={upper_5:.4f} → {status}")

    return uecm, bt, cointegrated, status


# ============================================================================
# 8. ECM INTERPRETATION
# ============================================================================
def interpret_ecm(uecm_fit, cointegrated, verbose=True):
    if verbose:
        print("\n[6/8] Error Correction Model (UECM) Diagnostics")
    if not cointegrated:
        if verbose:
            print("      Tidak ada kointegrasi — ECM tidak diinterpretasikan.")
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
        if verbose:
            print("      Tidak menemukan koefisien speed-of-adjustment.")
        return None

    lam = params[lambda_name]
    t_lam = tvalues[lambda_name]
    p_lam = pvalues[lambda_name]

    if verbose:
        print(f"      Speed-of-adjustment (λ) on {lambda_name}: {lam:.6f}")
        print(f"      t-stat: {t_lam:.4f}   p-value: {p_lam:.4f}")
        if lam < 0 and p_lam < 0.05:
            print("      Sign & signifikansi: OK (λ<0, p<0.05)")
        elif lam < 0:
            print("      λ negatif tapi tidak signifikan pada 5%.")
        else:
            print("      PERINGATAN: λ tidak negatif.")

    half_life = np.log(0.5) / np.log(1 + lam) if -1 < lam < 0 else np.nan
    if verbose and np.isfinite(half_life):
        print(f"      Half-life menuju equilibrium: ~{half_life:.2f} bulan")

    # ---- Long-run coefficients ----
    # Dalam UECM statsmodels, level terms adalah params tanpa prefix "D." dan bukan lambda
    # Format: "log_Open.L1", "log_Close.L1", "log_Volume.L1"
    lr_coefs = {}
    level_params = {k: v for k, v in params.items()
                    if not k.startswith("D.") and k != lambda_name and k != "const"}
    if verbose and level_params:
        print(f"\n      Long-run Coefficients (β_LR = -coef / λ):")
        print(f"      {'Variabel':<20} {'UECM coef':>12} {'LR Coef':>12} {'t-stat':>10} {'p-value':>10}")
        print("      " + "-" * 68)
    for name, coef in level_params.items():
        lr = -coef / lam if lam != 0 else np.nan
        t_lr = float(tvalues.get(name, np.nan))
        p_lr = float(pvalues.get(name, np.nan))
        var_label = name.split(".")[0]  # "log_Open.L1" → "log_Open"
        lr_coefs[var_label] = {"uecm_coef": coef, "lr_coef": lr, "t": t_lr, "p": p_lr}
        if verbose:
            print(f"      {var_label:<20} {coef:>12.6f} {lr:>12.6f} {t_lr:>10.4f} {p_lr:>10.4f}")

    # ---- Residual Diagnostics ----
    resid = uecm_fit.resid
    result = {"lambda": lam, "t": t_lam, "p": p_lam, "half_life_months": half_life,
              "long_run": lr_coefs}

    if verbose:
        print(f"\n      Residual Diagnostics:")

    # 1. Breusch-Godfrey serial correlation (lag=4)
    # BG requires exog aligned to resid length — use resid directly via manual LM
    try:
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm
        n_resid = len(resid)
        nlags = 4
        resid_arr = resid.values if hasattr(resid, "values") else np.array(resid)
        # Build lag matrix of residuals
        lag_mat = np.column_stack([resid_arr[nlags - j: n_resid - j] for j in range(1, nlags + 1)])
        y_bg = resid_arr[nlags:]
        X_bg = sm.add_constant(lag_mat)
        bg_fit = OLS(y_bg, X_bg).fit()
        bg_lm = len(y_bg) * bg_fit.rsquared
        from scipy import stats as scipy_stats
        bg_p = float(scipy_stats.chi2.sf(bg_lm, df=nlags))
        result["bg_lm"] = bg_lm
        result["bg_p"] = bg_p
        if verbose:
            tag = "OK (tidak ada serial korelasi)" if bg_p > 0.05 else "PERINGATAN: ada serial korelasi"
            print(f"      BG Serial Corr (lag=4) : LM={bg_lm:.4f}, p={bg_p:.4f}  → {tag}")
    except Exception as e:
        result["bg_p"] = np.nan
        if verbose:
            print(f"      BG Serial Corr         : tidak dapat dihitung ({e})")

    # 2. ARCH LM heteroskedasticity (lag=4)
    try:
        arch_lm, arch_p, _, _ = het_arch(resid, nlags=4)
        result["arch_lm"] = arch_lm
        result["arch_p"] = arch_p
        if verbose:
            tag = "OK (tidak ada ARCH effect)" if arch_p > 0.05 else "PERINGATAN: ada heteroskedastisitas kondisional"
            print(f"      ARCH LM (lag=4)        : LM={arch_lm:.4f}, p={arch_p:.4f}  → {tag}")
    except Exception:
        result["arch_p"] = np.nan

    # 3. Jarque-Bera normality
    try:
        jb_stat, jb_p, skew, kurt = jarque_bera(resid)
        result["jb_stat"] = jb_stat
        result["jb_p"] = jb_p
        if verbose:
            tag = "OK (normal)" if jb_p > 0.05 else "Non-normal (robust SE tetap valid untuk n besar)"
            print(f"      Jarque-Bera Normality  : JB={jb_stat:.4f}, p={jb_p:.4f}  → {tag}")
            print(f"        Skewness={skew:.4f}, Kurtosis={kurt:.4f}")
    except Exception:
        result["jb_p"] = np.nan

    # 4. CUSUM structural stability (manual — statsmodels returns 3-tuple)
    try:
        cusum_result = breaks_cusumolsresid(resid)
        cusum_arr = cusum_result[0]
        cusum_crit = cusum_result[1]
        cusum_stable = bool(np.all(np.abs(cusum_arr) <= cusum_crit))
        result["cusum_stable"] = cusum_stable
        if verbose:
            tag = "STABIL (tidak ada structural break)" if cusum_stable else "PERINGATAN: indikasi structural break"
            print(f"      CUSUM Stability        : {tag}")
    except Exception as e:
        result["cusum_stable"] = None
        if verbose:
            print(f"      CUSUM Stability        : tidak dapat dihitung ({e})")

    return result


# ============================================================================
# 9. EXOG FORECAST via VAR (monthly)
# ============================================================================
def forecast_exog_var(exog, horizon=1, verbose=True):
    var_fit = VAR(exog).fit(maxlags=VAR_MAXLAG, ic="aic")
    p = var_fit.k_ar
    if p == 0:
        # statsmodels bisa return k_ar=0 kalau AIC pilih no lag — fallback ke 1
        var_fit = VAR(exog).fit(1)
        p = 1
    seed = exog.values[-p:]
    fc = var_fit.forecast(seed, steps=horizon)

    last_date = exog.index[-1]
    future_idx = pd.date_range(start=last_date + pd.offsets.MonthEnd(1),
                               periods=horizon, freq="M")
    exog_future = pd.DataFrame(fc, index=future_idx, columns=exog.columns)
    if verbose:
        mode_label = "log-scale" if USE_LOG_TRANSFORM else "level"
        print(f"\n      [Exog projection] VAR(p) pada {list(exog.columns)} ({mode_label})")
        print(f"      VAR order terpilih (AIC): {p} lag")
        print(f"      Exog diproyeksikan {horizon} bulan: {future_idx[0].strftime('%Y-%m')} → {future_idx[-1].strftime('%Y-%m')}")
    return exog_future


# ============================================================================
# 10. MONTHLY FORECAST (1-step ahead)
# ============================================================================
def forecast_monthly(fit, endog, exog_future, horizon=1, verbose=True):
    n = len(endog)
    pred = fit.get_prediction(start=n, end=n + horizon - 1, exog_oos=exog_future)
    frame = pred.summary_frame(alpha=0.05)

    col_mean = "mean" if "mean" in frame.columns else frame.columns[0]
    col_lo = next((c for c in frame.columns if "lower" in c.lower()), None)
    col_hi = next((c for c in frame.columns if "upper" in c.lower()), None)

    # Bias correction: σ² dari residual training (log scale)
    sigma2 = float(np.var(fit.resid)) if USE_LOG_TRANSFORM else 0.0

    # inverse transform ke level Rupiah dengan bias correction exp(ŷ + σ²/2)
    out = pd.DataFrame({
        "Predicted_Low": to_level(frame[col_mean].values, sigma2),
        "CI_Lower": to_level(frame[col_lo].values, sigma2) if col_lo else np.nan,
        "CI_Upper": to_level(frame[col_hi].values, sigma2) if col_hi else np.nan,
    }, index=exog_future.index)

    if verbose:
        print(f"\n[7/8] Forecast Monthly Low ({horizon} bulan ke depan)")
        for dt, row in out.iterrows():
            print(f"      {dt.strftime('%Y-%m')}: Predicted=Rp {row['Predicted_Low']:,.0f}  "
                  f"CI=[Rp {row['CI_Lower']:,.0f} – Rp {row['CI_Upper']:,.0f}]")
    return out


# ============================================================================
# 11. ROLLING BACKTEST (walk-forward)
# ============================================================================
def rolling_backtest(monthly, n_backtest_months=12):
    print(f"\n{'='*72}")
    print(f"  ROLLING BACKTEST — {n_backtest_months} bulan terakhir (walk-forward)")
    print(f"{'='*72}")

    results = []
    total = len(monthly)
    if total < n_backtest_months + MAX_LAG_ENDOG + 12:
        raise RuntimeError(f"Data tidak cukup untuk backtest {n_backtest_months} bulan.")

    for i in range(n_backtest_months, 0, -1):
        train_end_idx = total - i
        train = monthly.iloc[:train_end_idx]
        actual_row = monthly.iloc[train_end_idx]
        target_dt = actual_row.name

        try:
            train = apply_rolling_window(train)
            endog_t, exog_t = prepare_variables(train)
            endog_lag, exog_orders = select_lag_order(endog_t, exog_t, verbose=False)
            fit = estimate_ardl(endog_t, exog_t, endog_lag, exog_orders, verbose=False)
            _, bt, cointegrated, status = run_bounds_test(endog_t, exog_t, endog_lag, exog_orders, verbose=False)
            exog_fut = forecast_exog_var(exog_t, horizon=1, verbose=False)
            forecast = forecast_monthly(fit, endog_t, exog_fut, horizon=1, verbose=False)

            pred_row = forecast.iloc[0]
            pred_low = pred_row["Predicted_Low"]
            ci_lo = pred_row["CI_Lower"]
            ci_hi = pred_row["CI_Upper"]
            actual_low = actual_row["Low"]
            err_pct = abs(actual_low - pred_low) / actual_low * 100
            in_ci = bool(ci_lo <= actual_low <= ci_hi)

            print(f"  [{n_backtest_months - i + 1:2d}/{n_backtest_months}] {target_dt.strftime('%Y-%m')} "
                  f"actual=Rp {actual_low:>14,.0f}  pred=Rp {pred_low:>14,.0f}  "
                  f"err={err_pct:>5.2f}%  CI={'Y' if in_ci else 'N'}  "
                  f"F={bt.stat:>6.2f}  status={status}")

            results.append({
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
            })
        except Exception as e:
            print(f"  [{n_backtest_months - i + 1:2d}/{n_backtest_months}] {target_dt.strftime('%Y-%m')} GAGAL: {e}")
            results.append({
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
            })

    return pd.DataFrame(results)


def evaluate_backtest(results_df):
    print(f"\n{'='*72}")
    print(f"  EVALUASI ROLLING BACKTEST")
    print(f"{'='*72}")

    valid = results_df.dropna(subset=["predicted_low"])
    if valid.empty:
        print("  Semua iterasi gagal — tidak ada metrik.")
        return None

    actual = valid["actual_low"].values
    predicted = valid["predicted_low"].values
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    r2 = r2_score(actual, predicted)
    ci_coverage = valid["in_ci"].sum() / len(valid) * 100
    coint_rate = (valid["cointegration"] == "COINTEGRATED").sum() / len(valid) * 100

    print(f"\n  Sample size  : {len(valid)} bulan")
    print(f"  MAPE         : {mape:.2f}%", end="")
    if mape < 5:    print("  [EXCELLENT]")
    elif mape < 10: print("  [GOOD]")
    elif mape < 15: print("  [ACCEPTABLE]")
    else:           print("  [PERLU DITINGKATKAN]")
    print(f"  RMSE         : Rp {rmse:,.0f}")
    print(f"  MAE          : Rp {mae:,.0f}")
    print(f"  R²           : {r2:.4f}")
    print(f"  CI Coverage  : {ci_coverage:.1f}% ({valid['in_ci'].sum()}/{len(valid)} dalam 95% CI)")
    print(f"  Cointegrated : {coint_rate:.1f}% iterasi (sisanya inconclusive/none)")

    return {
        "mape": mape, "rmse": rmse, "mae": mae, "r2": r2,
        "ci_coverage": ci_coverage, "n": len(valid), "coint_rate": coint_rate,
    }


# ============================================================================
# 12. SINGLE-MONTH EVALUASI vs Aktual
# ============================================================================
def evaluate_single(forecast_df, eval_to_date):
    """Untuk MODE='single': bandingkan 1 bulan forecast dengan aktual."""
    print(f"\n{'='*72}")
    print(f"  EVALUASI: Forecast vs Aktual ({forecast_df.index[0].strftime('%Y-%m')})")
    print(f"{'='*72}")

    actual_daily = fetch_btc_daily(to_date=eval_to_date)
    actual_monthly = resample_to_monthly(actual_daily, drop_partial=False)

    forecast_month = forecast_df.index[0]
    target_period = actual_monthly.index[
        (actual_monthly.index.year == forecast_month.year) &
        (actual_monthly.index.month == forecast_month.month)
    ]

    if target_period.empty:
        print("      Tidak ada data aktual untuk bulan target.")
        return None

    actual_low = actual_monthly.loc[target_period[0], "Low"]
    pred_row = forecast_df.iloc[0]

    err_rp = pred_row["Predicted_Low"] - actual_low
    err_pct = abs(err_rp) / actual_low * 100
    in_ci = bool(pred_row["CI_Lower"] <= actual_low <= pred_row["CI_Upper"])

    print(f"\n  Bulan target : {forecast_month.strftime('%Y-%m')}")
    print(f"  Actual Low   : Rp {actual_low:,.0f}")
    print(f"  Predicted    : Rp {pred_row['Predicted_Low']:,.0f}")
    print(f"  CI 95%       : [Rp {pred_row['CI_Lower']:,.0f} – Rp {pred_row['CI_Upper']:,.0f}]")
    print(f"  Error        : Rp {err_rp:+,.0f} ({err_pct:.2f}%)")
    print(f"  Dalam CI     : {'Ya' if in_ci else 'Tidak'}")

    return {
        "actual_low": actual_low,
        "predicted_low": pred_row["Predicted_Low"],
        "ci_lower": pred_row["CI_Lower"],
        "ci_upper": pred_row["CI_Upper"],
        "error_rp": err_rp,
        "error_pct": err_pct,
        "in_ci": in_ci,
    }


# ============================================================================
# 13. OUTPUT
# ============================================================================
def save_backtest_outputs(results_df, metrics, monthly_history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_month = results_df["month"].iloc[-1]
    tag = last_month.strftime("%Y%m")

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, f"low_ardl_monthly_backtest_{tag}.csv")
    out = results_df.copy()
    out["month"] = out["month"].dt.strftime("%Y-%m")
    out["actual_low"] = out["actual_low"].apply(lambda x: f"Rp {x:,.0f}")
    out["predicted_low"] = out["predicted_low"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["ci_lower"] = out["ci_lower"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["ci_upper"] = out["ci_upper"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["error_pct"] = out["error_pct"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    out["bounds_F"] = out["bounds_F"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    out.to_csv(csv_path, index=False)
    print(f"\n      Backtest CSV: {csv_path}")

    # Chart (skip kalau semua iterasi gagal)
    if metrics is not None:
        chart_path = os.path.join(OUTPUT_DIR, f"low_ardl_monthly_backtest_chart_{tag}.png")
        plot_backtest(results_df, monthly_history, metrics, chart_path)
        print(f"      Chart PNG   : {chart_path}")
    else:
        print("      Chart PNG   : dilewati (tidak ada iterasi sukses)")


def save_single_outputs(forecast_df, eval_result, monthly_history, ardl_info):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_month = forecast_df.index[0]
    tag = target_month.strftime("%Y%m")

    csv_path = os.path.join(OUTPUT_DIR, f"low_ardl_monthly_{tag}.csv")
    pred_row = forecast_df.iloc[0]
    row = {
        "target_month": target_month.strftime("%Y-%m"),
        "predicted_low": f"Rp {pred_row['Predicted_Low']:,.0f}",
        "ci_lower": f"Rp {pred_row['CI_Lower']:,.0f}",
        "ci_upper": f"Rp {pred_row['CI_Upper']:,.0f}",
        "endog_lag": ardl_info["endog_lag"],
        "exog_orders": str(ardl_info["exog_orders"]),
        "bounds_F": f"{ardl_info['bounds_F']:.4f}",
        "cointegration": ardl_info["cointegration"],
        "lambda_ecm": f"{ardl_info['lambda']:.6f}" if ardl_info.get("lambda") is not None else "N/A",
        "rolling_window_years": ROLLING_WINDOW_YEARS if ROLLING_WINDOW_YEARS > 0 else "all",
        "log_transform": USE_LOG_TRANSFORM,
    }
    if eval_result:
        row.update({
            "actual_low": f"Rp {eval_result['actual_low']:,.0f}",
            "error_pct": f"{eval_result['error_pct']:.2f}%",
            "in_ci": "Ya" if eval_result["in_ci"] else "Tidak",
        })
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f"\n      Monthly CSV : {csv_path}")

    chart_path = os.path.join(OUTPUT_DIR, f"low_ardl_monthly_chart_{tag}.png")
    plot_single(forecast_df, monthly_history, eval_result, chart_path)
    print(f"      Chart PNG   : {chart_path}")


def plot_backtest(results_df, monthly_history, metrics, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))

    # Panel 1: Actual vs Predicted line + CI band
    ax1 = axes[0]
    valid = results_df.dropna(subset=["predicted_low"]).copy()
    months_str = [m.strftime("%Y-%m") for m in valid["month"]]
    ax1.plot(months_str, valid["actual_low"] / 1e9, "o-", color="#4CAF50",
             linewidth=2, markersize=7, label="Aktual")
    ax1.plot(months_str, valid["predicted_low"] / 1e9, "s--", color="#E91E63",
             linewidth=2, markersize=7, label="Prediksi")
    ax1.fill_between(months_str, valid["ci_lower"] / 1e9, valid["ci_upper"] / 1e9,
                     color="#E91E63", alpha=0.15, label="95% CI")
    ax1.set_title(f"Rolling Backtest — Actual vs Predicted Monthly Low (n={metrics['n']})",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Low (Miliar IDR)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Panel 2: Error % per bulan
    ax2 = axes[1]
    err = ((valid["predicted_low"] - valid["actual_low"]) / valid["actual_low"] * 100).values
    colors = ["#4CAF50" if e >= 0 else "#F44336" for e in err]
    ax2.bar(months_str, err, color=colors, alpha=0.75, edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axhline(y=err.mean(), color="#FF9800", linestyle="--",
                linewidth=2, label=f"Mean error: {err.mean():.2f}%")
    ax2.set_title(f"Error Bulanan (MAPE rata-rata: {metrics['mape']:.2f}%)",
                  fontsize=13, fontweight="bold")
    ax2.set_ylabel("Error (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=45)

    # Panel 3: CI coverage map
    ax3 = axes[2]
    in_ci = valid["in_ci"].astype(int).values
    colors_ci = ["#4CAF50" if x else "#F44336" for x in in_ci]
    ax3.bar(months_str, [1] * len(months_str), color=colors_ci, alpha=0.85, edgecolor="white")
    ax3.set_yticks([])
    ax3.set_title(f"CI Coverage: {metrics['ci_coverage']:.1f}% ({valid['in_ci'].sum()}/{len(valid)} dalam 95% CI)  "
                  f"[hijau=in CI, merah=miss]", fontsize=13, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)

    # Panel 4: Bounds F-stat per iterasi
    ax4 = axes[3]
    f_vals = valid["bounds_F"].values
    ax4.plot(months_str, f_vals, "o-", color="#1976D2", linewidth=2, markersize=7)
    ax4.axhline(y=4.0, color="#F44336", linestyle="--", linewidth=1.5,
                label="approx I(1) upper @5% (3 regressors)")
    ax4.set_title(f"Bounds F-stat per Iterasi  ({metrics['coint_rate']:.0f}% iter cointegrated)",
                  fontsize=13, fontweight="bold")
    ax4.set_ylabel("F-statistic")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_single(forecast_df, monthly_history, eval_result, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    recent = monthly_history.tail(36)
    months_hist = [m.strftime("%Y-%m") for m in recent.index]
    fc_label = forecast_df.index[0].strftime("%Y-%m")

    ax.plot(months_hist, recent["Low"] / 1e9, "o-", color="#2196F3",
            linewidth=1.5, markersize=5, label="Historis Monthly Low")
    fc_val = forecast_df.iloc[0]["Predicted_Low"] / 1e9
    ax.errorbar([fc_label], [fc_val],
                yerr=[[fc_val - forecast_df.iloc[0]["CI_Lower"] / 1e9],
                      [forecast_df.iloc[0]["CI_Upper"] / 1e9 - fc_val]],
                fmt="*", color="#E91E63", markersize=18, capsize=8,
                label=f"Forecast {fc_label}")
    if eval_result:
        ax.plot([fc_label], [eval_result["actual_low"] / 1e9], "D",
                color="#4CAF50", markersize=12,
                label=f"Aktual {fc_label}: Rp {eval_result['actual_low']:,.0f}")

    ax.set_title(f"Monthly ARDL-ECM Forecast — {fc_label}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Low (Miliar IDR)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def run_single():
    """MODE='single' atau 'forecast_only'."""
    print("\n[1/8] Data Acquisition")
    daily = fetch_btc_daily(to_date=TRAIN_END)
    monthly_full = resample_to_monthly(daily)
    monthly = apply_rolling_window(monthly_full)
    if ROLLING_WINDOW_YEARS > 0:
        print(f"      Rolling window: {ROLLING_WINDOW_YEARS} tahun terakhir → {len(monthly)} bulan")
    log_label = "log-scale (USE_LOG_TRANSFORM=True)" if USE_LOG_TRANSFORM else "level (USE_LOG_TRANSFORM=False)"
    print(f"      Transform: {log_label}")

    endog, exog = prepare_variables(monthly)
    check_stationarity(endog, exog)
    endog_lag, exog_orders = select_lag_order(endog, exog)
    fit = estimate_ardl(endog, exog, endog_lag, exog_orders)
    uecm_fit, bt, cointegrated, status = run_bounds_test(endog, exog, endog_lag, exog_orders)
    ecm_info = interpret_ecm(uecm_fit, cointegrated)

    print(f"\n      Forecast horizon: 1 bulan ke depan")
    exog_future = forecast_exog_var(exog, horizon=1)
    forecast = forecast_monthly(fit, endog, exog_future, horizon=1)

    eval_result = None
    if MODE == "single":
        eval_result = evaluate_single(forecast, EVAL_END)

    ardl_info = {
        "endog_lag": endog_lag,
        "exog_orders": exog_orders,
        "bounds_F": bt.stat,
        "cointegration": status,
        "lambda": ecm_info["lambda"] if ecm_info else None,
    }
    save_single_outputs(forecast, eval_result, monthly, ardl_info)

    print("\n" + "=" * 72)
    print("  SUMMARY (Single Monthly Forecast)")
    print("=" * 72)
    print(f"  Model         : ARDL(p={endog_lag}, orders={exog_orders})")
    print(f"  Cointegration : {status} (F={bt.stat:.4f})")
    if ecm_info:
        print(f"  ECM λ         : {ecm_info['lambda']:.6f} (p={ecm_info['p']:.4f})")
        if np.isfinite(ecm_info.get("half_life_months", np.nan)):
            print(f"  Half-life     : {ecm_info['half_life_months']:.2f} bulan")
    pred = forecast.iloc[0]
    print(f"  Target month  : {forecast.index[0].strftime('%Y-%m')}")
    print(f"  Predicted Low : Rp {pred['Predicted_Low']:,.0f}")
    print(f"  CI 95%        : [Rp {pred['CI_Lower']:,.0f} – Rp {pred['CI_Upper']:,.0f}]")
    if eval_result:
        print(f"\n  Evaluasi vs Aktual:")
        print(f"    Actual      : Rp {eval_result['actual_low']:,.0f}")
        print(f"    Error %     : {eval_result['error_pct']:.2f}%")
        print(f"    Dalam CI    : {'Ya' if eval_result['in_ci'] else 'Tidak'}")
    print("=" * 72)


def run_backtest():
    """MODE='rolling_backtest'."""
    print("\n[1/8] Data Acquisition")
    daily = fetch_btc_daily(to_date=TRAIN_END)
    monthly = resample_to_monthly(daily)

    log_label = "log-scale" if USE_LOG_TRANSFORM else "level"
    win_label = f"{ROLLING_WINDOW_YEARS} tahun terakhir per iterasi" if ROLLING_WINDOW_YEARS > 0 else "semua data"
    print(f"      Transform: {log_label} | Window training: {win_label}")

    # ADF pada full monthly pakai setting transform saat ini (untuk info)
    endog_full, exog_full = prepare_variables(monthly)
    check_stationarity(endog_full, exog_full)

    print(f"\n      Mulai walk-forward backtest {BACKTEST_MONTHS} bulan...")
    print(f"      Per iterasi: rolling window + refit lag + ARDL + Bounds + VAR + 1-step forecast")

    results = rolling_backtest(monthly, n_backtest_months=BACKTEST_MONTHS)
    metrics = evaluate_backtest(results)
    save_backtest_outputs(results, metrics, monthly)

    print("\n" + "=" * 72)
    print(f"  SUMMARY (Rolling Backtest {BACKTEST_MONTHS} bulan)")
    print("=" * 72)
    if metrics:
        print(f"  Sample        : {metrics['n']} bulan")
        print(f"  MAPE          : {metrics['mape']:.2f}%")
        print(f"  RMSE          : Rp {metrics['rmse']:,.0f}")
        print(f"  MAE           : Rp {metrics['mae']:,.0f}")
        print(f"  R²            : {metrics['r2']:.4f}")
        print(f"  CI Coverage   : {metrics['ci_coverage']:.1f}%")
        print(f"  Cointegrated  : {metrics['coint_rate']:.0f}% iterasi")
    print("=" * 72)


def main():
    print("=" * 72)
    print("  BTC/IDR MONTHLY LOW — ARDL-ECM (Econometric, Monthly Frequency)")
    print(f"  Mode          : {MODE}")
    print(f"  Training s/d  : {TRAIN_END}")
    win_str = f"{ROLLING_WINDOW_YEARS} tahun terakhir" if ROLLING_WINDOW_YEARS > 0 else "semua data (sejak 2014)"
    print(f"  Training window: {win_str}")
    print(f"  Log-transform : {'Ya (log Y & X)' if USE_LOG_TRANSFORM else 'Tidak (level)'}")
    if MODE == "rolling_backtest":
        print(f"  Backtest      : {BACKTEST_MONTHS} bulan terakhir (walk-forward)")
    elif MODE == "single":
        print(f"  Evaluasi s/d  : {EVAL_END}")
    print("=" * 72)

    if MODE == "rolling_backtest":
        run_backtest()
    elif MODE in ("single", "forecast_only"):
        run_single()
    else:
        raise ValueError(f"MODE tidak dikenali: {MODE}")


if __name__ == "__main__":
    main()
