"""
BTC/IDR Daily Low Prediction — ARDL-ECM (Econometric)
======================================================
Implementasi metodologi ARDL yang benar sesuai panduan.md:
    Step 1: Definisi variabel  (Y=Low, X=[Open, Close, log_Volume])
    Step 2: ADF stationarity test (level + first difference)
    Step 3: Lag selection via AIC (statsmodels.ardl_select_order)
    Step 4: ARDL estimation via OLS (statsmodels.tsa.ardl.ARDL)
    Step 5: Bounds Test untuk kointegrasi (Pesaran-Shin-Smith 2001, via UECM)
    Step 6: Error Correction Model (UECM) — speed-of-adjustment lambda
    Step 7: Dynamic forecasting (exog diproyeksikan via VAR, bukan proxy sintetik)
    Step 8: Monthly low = min() dari forecast harian bulan target

Cara jalankan:
    python btc_low_ardl_ecm.py

Mode:
    MODE = "forecast_only"        — train s/d TRAIN_END, forecast bulan depan
    MODE = "forecast_and_evaluate"— train s/d TRAIN_END, forecast FORECAST_MONTH,
                                    fetch aktual s/d EVAL_END, bandingkan MAPE/RMSE
    MODE = "rolling_backtest"     — walk-forward N bulan terakhir, refit per iterasi
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
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# KONFIGURASI
# ============================================================================
MODE = "forecast_and_evaluate"  # "forecast_only" | "forecast_and_evaluate" | "rolling_backtest"
TRAIN_END = "2025-12-31"        # Data training berakhir di sini
EVAL_END = "2026-01-31"         # Untuk evaluasi: fetch aktual sampai sini

MAX_LAG_ENDOG = 5
MAX_LAG_EXOG = 5
IC = "aic"
TREND = "c"

ROLLING_WINDOW_YEARS = 0      # 0 = semua data sejak 2014; >0 = N tahun terakhir per iterasi
USE_LOG_TRANSFORM = False     # True = log(Y) & log(X), output di-inverse ke Rupiah
BACKTEST_MONTHS = 12          # jumlah bulan untuk rolling_backtest mode

OUTPUT_DIR = "output"

API_URL = "https://indodax.com/tradingview/history_v2"
SYMBOL = "BTCIDR"
TIMEFRAME = "1D"
FROM_TS = 1391191321  # 2014-02-01


# ============================================================================
# HELPERS
# ============================================================================
def to_level(series):
    return np.exp(series) if USE_LOG_TRANSFORM else series

def apply_rolling_window(df):
    if ROLLING_WINDOW_YEARS > 0:
        cutoff = df.index[-1] - pd.DateOffset(years=ROLLING_WINDOW_YEARS)
        return df[df.index >= cutoff]
    return df


# ============================================================================
# 1. DATA FETCH
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

    print(f"      Data bersih: {len(df)} records ({df.index.min().date()} — {df.index.max().date()})")
    return df


# ============================================================================
# 2. PREPARE VARIABLES (Step 1 panduan)
# ============================================================================
def prepare_variables(df):
    if USE_LOG_TRANSFORM:
        endog = np.log(df["Low"]).copy()
        exog = pd.DataFrame({
            "log_Open":   np.log(df["Open"]),
            "log_Close":  np.log(df["Close"]),
            "log_Volume": np.log1p(df["Volume"]),
        }, index=df.index)
    else:
        endog = df["Low"].copy()
        exog = pd.DataFrame({
            "Open": df["Open"],
            "Close": df["Close"],
            "log_Volume": np.log1p(df["Volume"]),
        }, index=df.index)
    return endog, exog


# ============================================================================
# 3. ADF STATIONARITY TEST (Step 2 panduan)
# ============================================================================
def check_stationarity(endog, exog, verbose=True):
    if verbose:
        print("\n[2/8] ADF Stationarity Test")
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
            if p_val < 0.05:
                tag = "I(1) OK"
                results[name]["order"] = 1
            else:
                tag = "I(2) WARNING — ARDL tidak valid"
                results[name]["order"] = 2
            if verbose:
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
# 4. LAG SELECTION (Step 3 panduan)
# ============================================================================
def select_lag_order(endog, exog, verbose=True):
    if verbose:
        print(f"\n[3/8] Lag Selection (ic={IC}, maxlag={MAX_LAG_ENDOG}, maxorder={MAX_LAG_EXOG})")
        print("      Mencari kombinasi lag terbaik...")

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
    exog_cols = list(exog.columns)
    exog_orders = dict(zip(exog_cols, ardl_order[1:]))

    if verbose:
        print(f"      Best p (endog lag Low): {endog_lag}")
        for k, v in exog_orders.items():
            print(f"      Best q ({k}): {v}")
    return endog_lag, exog_orders


# ============================================================================
# 5. ARDL ESTIMATION (Step 4 panduan)
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
        print("      Top 8 coefficients:")
        for name, val in list(fit.params.items())[:8]:
            print(f"        {name:<30} {val:>16,.4f}")
    return fit


# ============================================================================
# 6. BOUNDS TEST (Step 5 panduan)
# ============================================================================
def run_bounds_test(endog, exog, endog_lag, exog_orders, verbose=True):
    if verbose:
        print("\n[5/8] Bounds Test untuk Kointegrasi (Pesaran-Shin-Smith)")

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
        print(f"      F-statistic: {bt.stat:.4f}")
        print(f"      p-value    : {bt.p_values['upper']:.4f}  (pakai upper-bound)")
        print("      Critical Values:")
        print(f"      {'Level':<8} {'I(0) Lower':>14} {'I(1) Upper':>14}")
        for level in bt.crit_vals.index:
            lo = bt.crit_vals.loc[level, "lower"]
            up = bt.crit_vals.loc[level, "upper"]
            print(f"      {str(level):<8} {lo:>14.4f} {up:>14.4f}")
        if f_stat > upper_5:
            print(f"\n      Keputusan @5%: F={f_stat:.4f} > I(1) upper={upper_5:.4f} → {status}")
            print("      Gunakan ECM (ada hubungan jangka panjang).")
        elif f_stat < lower_5:
            print(f"\n      Keputusan @5%: F={f_stat:.4f} < I(0) lower={lower_5:.4f} → {status}")
            print("      Tidak ada hubungan jangka panjang — gunakan ARDL dalam first-difference.")
        else:
            print(f"\n      Keputusan @5%: {lower_5:.4f} < F={f_stat:.4f} < {upper_5:.4f} → {status}")
            print("      Inconclusive — default ke jalur ECM dengan catatan kehati-hatian.")

    return uecm, bt, cointegrated, status


# ============================================================================
# 7. ECM INTERPRETATION (Step 6 panduan)
# ============================================================================
def interpret_ecm(uecm_fit, cointegrated, verbose=True):
    if verbose:
        print("\n[6/8] Error Correction Model (UECM) Diagnostics")
    if not cointegrated:
        if verbose:
            print("      Tidak ada kointegrasi — ECM tidak diinterpretasikan.")
            print("      Estimasi dynamics dari ARDL dalam first-difference direkomendasikan.")
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
            if "L1" in name and name.startswith(("Low", "y", "log")):
                lambda_name = name
                break

    if lambda_name is None:
        if verbose:
            print("      Tidak menemukan koefisien speed-of-adjustment di UECM params.")
            print("      Params tersedia (5 pertama):", list(params.index[:5]))
        return None

    lam = params[lambda_name]
    t_lam = tvalues[lambda_name]
    p_lam = pvalues[lambda_name]

    if verbose:
        print(f"      Speed-of-adjustment (λ) on {lambda_name}: {lam:.6f}")
        print(f"      t-stat: {t_lam:.4f}   p-value: {p_lam:.4f}")
        if lam < 0 and p_lam < 0.05:
            print("      Sign & signifikansi: OK (λ<0, p<0.05) — mekanisme koreksi kesalahan valid.")
        elif lam < 0:
            print("      λ negatif tapi tidak signifikan pada 5%.")
        else:
            print("      PERINGATAN: λ tidak negatif — ECM tidak berfungsi sebagaimana mestinya.")

    half_life = np.log(0.5) / np.log(1 + lam) if -1 < lam < 0 else np.nan
    if verbose and np.isfinite(half_life):
        print(f"      Half-life menuju equilibrium: ~{half_life:.2f} hari")
    return {"lambda": lam, "t": t_lam, "p": p_lam}


# ============================================================================
# 8. EXOG FORECAST via VAR
# ============================================================================
def forecast_exog_var(exog, horizon, verbose=True):
    if verbose:
        print(f"\n      [Exog projection] VAR(p) pada {list(exog.columns)}")
    var_fit = VAR(exog).fit(maxlags=5, ic="aic")
    p = var_fit.k_ar
    if p == 0:
        var_fit = VAR(exog).fit(1)
        p = 1
    seed = exog.values[-p:]
    fc = var_fit.forecast(seed, steps=horizon)

    future_idx = pd.date_range(start=exog.index[-1] + pd.Timedelta(days=1),
                               periods=horizon, freq="D")
    exog_future = pd.DataFrame(fc, index=future_idx, columns=exog.columns)
    if verbose:
        print(f"      VAR order terpilih (AIC): {p} lag")
        print(f"      Exog diproyeksikan {horizon} hari: {future_idx[0].date()} → {future_idx[-1].date()}")
        print("      Catatan: forecast ARDL bersifat CONDITIONAL pada projeksi VAR.")
    return exog_future


# ============================================================================
# 9. DAILY FORECAST (Step 7 panduan)
# ============================================================================
def forecast_daily(fit, endog, exog_future, horizon, verbose=True):
    if verbose:
        print(f"\n[7/8] Forecast harian ({horizon} hari ke depan)")
    n = len(endog)
    pred = fit.get_prediction(start=n, end=n + horizon - 1, exog_oos=exog_future)
    frame = pred.summary_frame(alpha=0.05)

    col_mean = "mean" if "mean" in frame.columns else frame.columns[0]
    col_lo = next((c for c in frame.columns if "lower" in c.lower()), None)
    col_hi = next((c for c in frame.columns if "upper" in c.lower()), None)

    out = pd.DataFrame({
        "Predicted_Low": to_level(frame[col_mean].values),
        "CI_Lower": to_level(frame[col_lo].values) if col_lo else np.nan,
        "CI_Upper": to_level(frame[col_hi].values) if col_hi else np.nan,
    }, index=exog_future.index)

    if verbose:
        print(f"      Forecast horizon: {out.index[0].date()} → {out.index[-1].date()}")
        print(f"      Rata-rata Predicted_Low: Rp {out['Predicted_Low'].mean():,.0f}")
        print(f"      Min Predicted_Low      : Rp {out['Predicted_Low'].min():,.0f}")
        print(f"      Max Predicted_Low      : Rp {out['Predicted_Low'].max():,.0f}")
    return out


# ============================================================================
# 10. DERIVE MONTHLY LOW (Step 8 panduan)
# ============================================================================
def derive_monthly_low(daily_forecast, target_month, verbose=True):
    mask = (daily_forecast.index.year == target_month.year) & \
           (daily_forecast.index.month == target_month.month)
    sub = daily_forecast.loc[mask]

    if sub.empty:
        raise RuntimeError(f"Tidak ada forecast untuk target month {target_month.strftime('%Y-%m')}")

    expected = sub["Predicted_Low"].min()
    worst = sub["CI_Lower"].min()
    best = sub["CI_Upper"].min()
    date_of_min = sub["Predicted_Low"].idxmin()

    if verbose:
        print(f"\n[8/8] Monthly Low Summary — {target_month.strftime('%Y-%m')}")
        print(f"      Expected monthly low: Rp {expected:,.0f}  (pada {date_of_min.date()})")
        print(f"      Worst-case (CI low) : Rp {worst:,.0f}")
        print(f"      Best-case (CI high) : Rp {best:,.0f}")
        print(f"      Hari yang di-forecast dalam bulan tsb: {len(sub)}")

    return {
        "target_month": target_month.strftime("%Y-%m"),
        "expected_low": expected,
        "worst_case_low": worst,
        "best_case_low": best,
        "date_of_min": date_of_min,
        "n_days": len(sub),
    }


# ============================================================================
# EVALUASI: Prediksi vs Aktual (forecast_and_evaluate mode)
# ============================================================================
def evaluate_forecast(daily_forecast, target_month, eval_to_date):
    print(f"\n{'='*72}")
    print(f"  EVALUASI: Prediksi vs Aktual ({target_month.strftime('%Y-%m')})")
    print(f"{'='*72}")

    actual_df = fetch_btc_daily(to_date=eval_to_date)
    mask = (actual_df.index.year == target_month.year) & \
           (actual_df.index.month == target_month.month)
    actual_period = actual_df.loc[mask, ["Low"]].copy()

    if actual_period.empty:
        print("      Tidak ada data aktual untuk periode ini.")
        return None, None

    merged = daily_forecast.join(actual_period.rename(columns={"Low": "Actual_Low"}), how="inner")
    merged = merged.dropna(subset=["Actual_Low"])

    if merged.empty:
        print("      Tidak ada overlap antara forecast dan data aktual.")
        return None, None

    actual = merged["Actual_Low"].values
    predicted = merged["Predicted_Low"].values

    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    r2 = r2_score(actual, predicted)
    in_ci = ((merged["Actual_Low"] >= merged["CI_Lower"]) &
             (merged["Actual_Low"] <= merged["CI_Upper"])).sum()
    ci_coverage = in_ci / len(merged) * 100

    print(f"\n  Metrik Evaluasi ({len(merged)} hari):")
    print(f"  MAPE : {mape:.2f}%", end="")
    if mape < 5:    print("  [EXCELLENT]")
    elif mape < 10: print("  [GOOD]")
    elif mape < 15: print("  [ACCEPTABLE]")
    else:           print("  [PERLU DITINGKATKAN]")
    print(f"  RMSE : Rp {rmse:,.0f}")
    print(f"  MAE  : Rp {mae:,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  CI Coverage: {ci_coverage:.1f}% ({in_ci}/{len(merged)} hari dalam 95% CI)")

    print(f"\n  {'Tanggal':<12} {'Aktual':>18} {'Prediksi':>18} {'Error%':>8} {'Dalam CI':<10}")
    print("  " + "-" * 72)
    for dt, row in merged.iterrows():
        err_pct = abs(row["Predicted_Low"] - row["Actual_Low"]) / row["Actual_Low"] * 100
        in_ci_flag = "Ya" if row["CI_Lower"] <= row["Actual_Low"] <= row["CI_Upper"] else "Tidak"
        print(f"  {dt.strftime('%Y-%m-%d'):<12} "
              f"Rp {row['Actual_Low']:>14,.0f} "
              f"Rp {row['Predicted_Low']:>14,.0f} "
              f"{err_pct:>7.2f}% "
              f"{in_ci_flag:<10}")

    metrics = {"mape": mape, "rmse": rmse, "mae": mae, "r2": r2, "ci_coverage": ci_coverage}
    return merged, metrics


# ============================================================================
# ROLLING BACKTEST (walk-forward, per-bulan)
# ============================================================================
def rolling_backtest_daily(daily, n_backtest_months=12):
    print(f"\n{'='*72}")
    print(f"  ROLLING BACKTEST — {n_backtest_months} bulan terakhir (walk-forward harian)")
    print(f"{'='*72}")

    results = []
    last_date = daily.index[-1]
    last_month_period = last_date.to_period("M")

    for i in range(n_backtest_months, 0, -1):
        target_period = last_month_period - i
        target_month = target_period.to_timestamp("M").replace(day=1)
        # cutoff: hari terakhir bulan sebelum target
        train_cutoff_dt = target_month - pd.Timedelta(days=1)
        train_cutoff_str = train_cutoff_dt.strftime("%Y-%m-%d")

        train = daily[daily.index <= train_cutoff_dt]
        train = apply_rolling_window(train)

        try:
            endog_t, exog_t = prepare_variables(train)
            endog_lag, exog_orders = select_lag_order(endog_t, exog_t, verbose=False)
            fit = estimate_ardl(endog_t, exog_t, endog_lag, exog_orders, verbose=False)
            _, bt, cointegrated, status = run_bounds_test(endog_t, exog_t, endog_lag, exog_orders, verbose=False)
            horizon, tgt_month = get_forecast_params(cutoff_date=train_cutoff_str)
            exog_future = forecast_exog_var(exog_t, horizon, verbose=False)
            daily_fc = forecast_daily(fit, endog_t, exog_future, horizon, verbose=False)
            monthly_summary = derive_monthly_low(daily_fc, tgt_month, verbose=False)

            # Ambil actual monthly low dari data harian
            mask = (daily.index.year == tgt_month.year) & (daily.index.month == tgt_month.month)
            actual_low = daily.loc[mask, "Low"].min() if mask.any() else np.nan
            pred_low = monthly_summary["expected_low"]
            err_pct = abs(actual_low - pred_low) / actual_low * 100 if pd.notna(actual_low) else np.nan
            in_ci = bool(monthly_summary["worst_case_low"] <= actual_low <= monthly_summary["best_case_low"]) \
                    if pd.notna(actual_low) else False

            print(f"  [{n_backtest_months - i + 1:2d}/{n_backtest_months}] {tgt_month.strftime('%Y-%m')} "
                  f"actual=Rp {actual_low:>14,.0f}  pred=Rp {pred_low:>14,.0f}  "
                  f"err={err_pct:>5.2f}%  CI={'Y' if in_ci else 'N'}  "
                  f"F={bt.stat:>6.2f}  status={status}")

            results.append({
                "month": tgt_month,
                "actual_low": actual_low,
                "predicted_low": pred_low,
                "ci_lower": monthly_summary["worst_case_low"],
                "ci_upper": monthly_summary["best_case_low"],
                "error_pct": err_pct,
                "in_ci": in_ci,
                "bounds_F": bt.stat,
                "cointegration": status,
                "endog_lag": endog_lag,
                "exog_orders": str(exog_orders),
            })
        except Exception as e:
            tgt_label = target_month.strftime("%Y-%m")
            print(f"  [{n_backtest_months - i + 1:2d}/{n_backtest_months}] {tgt_label} GAGAL: {e}")
            # Ambil actual low meski iterasi gagal
            mask = (daily.index.year == target_month.year) & (daily.index.month == target_month.month)
            actual_low = daily.loc[mask, "Low"].min() if mask.any() else np.nan
            results.append({
                "month": target_month,
                "actual_low": actual_low,
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
    print(f"  CI Coverage  : {ci_coverage:.1f}% ({valid['in_ci'].sum()}/{len(valid)} dalam CI)")
    print(f"  Cointegrated : {coint_rate:.1f}% iterasi")

    return {
        "mape": mape, "rmse": rmse, "mae": mae, "r2": r2,
        "ci_coverage": ci_coverage, "n": len(valid), "coint_rate": coint_rate,
    }


# ============================================================================
# 11. OUTPUT
# ============================================================================
def save_outputs(daily_forecast, monthly_summary, bounds_result, ecm_info,
                 cointegration_status, history_df, target_month, merged_eval=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = target_month.strftime("%Y%m")

    daily_csv = os.path.join(OUTPUT_DIR, f"low_ardl_ecm_daily_{tag}.csv")
    df_daily = daily_forecast.reset_index().rename(columns={"index": "Date", daily_forecast.index.name: "Date"})
    df_daily.columns = [c if c != df_daily.columns[0] else "Date" for c in df_daily.columns]
    df_daily.iloc[:, 0] = pd.to_datetime(df_daily.iloc[:, 0]).dt.strftime("%Y-%m-%d")
    df_daily["Predicted_Low"] = df_daily["Predicted_Low"].apply(lambda x: f"Rp {x:,.0f}")
    df_daily["CI_Lower"] = df_daily["CI_Lower"].apply(lambda x: f"Rp {x:,.0f}")
    df_daily["CI_Upper"] = df_daily["CI_Upper"].apply(lambda x: f"Rp {x:,.0f}")
    df_daily.to_csv(daily_csv, index=False)
    print(f"\n      Daily CSV  : {daily_csv}")

    monthly_csv = os.path.join(OUTPUT_DIR, f"low_ardl_ecm_monthly_{tag}.csv")
    row = {
        "target_month": monthly_summary["target_month"],
        "expected_low_rp": f"Rp {monthly_summary['expected_low']:,.0f}",
        "worst_case_low_rp": f"Rp {monthly_summary['worst_case_low']:,.0f}",
        "best_case_low_rp": f"Rp {monthly_summary['best_case_low']:,.0f}",
        "date_of_min": monthly_summary["date_of_min"].strftime("%Y-%m-%d"),
        "n_days_forecast": monthly_summary["n_days"],
        "bounds_F_stat": f"{bounds_result.stat:.4f}",
        "cointegration": cointegration_status,
        "lambda_ecm": f"{ecm_info['lambda']:.6f}" if ecm_info else "N/A",
        "lambda_pvalue": f"{ecm_info['p']:.4f}" if ecm_info else "N/A",
        "log_transform": USE_LOG_TRANSFORM,
        "rolling_window_years": ROLLING_WINDOW_YEARS if ROLLING_WINDOW_YEARS > 0 else "all",
    }
    pd.DataFrame([row]).to_csv(monthly_csv, index=False)
    print(f"      Monthly CSV: {monthly_csv}")

    chart_path = os.path.join(OUTPUT_DIR, f"low_ardl_ecm_chart_{tag}.png")
    plot_chart(history_df, daily_forecast, bounds_result, target_month,
               monthly_summary, chart_path, merged_eval)
    print(f"      Chart PNG  : {chart_path}")


def save_backtest_outputs(results_df, metrics, history_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    last_month = results_df["month"].iloc[-1]
    tag = last_month.strftime("%Y%m")

    csv_path = os.path.join(OUTPUT_DIR, f"low_ardl_ecm_backtest_{tag}.csv")
    out = results_df.copy()
    out["month"] = out["month"].dt.strftime("%Y-%m")
    out["actual_low"] = out["actual_low"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["predicted_low"] = out["predicted_low"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["ci_lower"] = out["ci_lower"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["ci_upper"] = out["ci_upper"].apply(lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A")
    out["error_pct"] = out["error_pct"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    out["bounds_F"] = out["bounds_F"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    out.to_csv(csv_path, index=False)
    print(f"\n      Backtest CSV: {csv_path}")

    if metrics is not None:
        chart_path = os.path.join(OUTPUT_DIR, f"low_ardl_ecm_backtest_chart_{tag}.png")
        plot_backtest(results_df, metrics, chart_path)
        print(f"      Chart PNG   : {chart_path}")
    else:
        print("      Chart PNG   : dilewati (tidak ada iterasi sukses)")


def plot_chart(history_df, daily_forecast, bounds_result, target_month,
               monthly_summary, save_path, merged_eval=None):
    has_eval = merged_eval is not None and len(merged_eval) > 0
    n_panels = 4 if has_eval else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels))

    ax1 = axes[0]
    recent = history_df.tail(90)
    ax1.plot(recent.index, recent["Low"] / 1e9, color="#2196F3", linewidth=1.5, label="Historis (Low)")
    ax1.plot(daily_forecast.index, daily_forecast["Predicted_Low"] / 1e9,
             color="#E91E63", linewidth=2, linestyle="--", label="Forecast ARDL-ECM")
    ax1.fill_between(daily_forecast.index,
                     daily_forecast["CI_Lower"] / 1e9,
                     daily_forecast["CI_Upper"] / 1e9,
                     color="#E91E63", alpha=0.15, label="95% CI")
    ax1.axvline(x=history_df.index[-1], color="gray", linestyle=":", linewidth=1.5, label="Batas data historis")
    if has_eval:
        ax1.plot(merged_eval.index, merged_eval["Actual_Low"] / 1e9,
                 color="#4CAF50", linewidth=1.5, marker="o", markersize=3, label="Aktual")
    ax1.scatter([monthly_summary["date_of_min"]],
                [monthly_summary["expected_low"] / 1e9],
                color="#FF9800", s=150, marker="*", zorder=5,
                label=f"Expected monthly low ({monthly_summary['target_month']})")
    ax1.set_title("BTC/IDR Daily Low — ARDL-ECM Forecast", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Harga Low (Miliar IDR)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    mask = (daily_forecast.index.year == target_month.year) & \
           (daily_forecast.index.month == target_month.month)
    zoom = daily_forecast.loc[mask]
    ax2.plot(zoom.index, zoom["Predicted_Low"] / 1e9,
             color="#E91E63", linewidth=2, marker="o", markersize=4, label="Forecast")
    ax2.fill_between(zoom.index, zoom["CI_Lower"] / 1e9, zoom["CI_Upper"] / 1e9,
                     color="#E91E63", alpha=0.2, label="95% CI")
    if has_eval:
        ax2.plot(merged_eval.index, merged_eval["Actual_Low"] / 1e9,
                 color="#4CAF50", linewidth=2, marker="s", markersize=4, label="Aktual")
    ax2.axhline(y=monthly_summary["expected_low"] / 1e9, color="#FF9800", linestyle="--",
                linewidth=1.5, label=f"Expected low: Rp {monthly_summary['expected_low']:,.0f}")
    ax2.axhline(y=monthly_summary["worst_case_low"] / 1e9, color="#F44336", linestyle=":",
                linewidth=1.5, label=f"Worst case: Rp {monthly_summary['worst_case_low']:,.0f}")
    ax2.set_title(f"Zoom: Forecast {target_month.strftime('%Y-%m')}", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Harga Low (Miliar IDR)")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    ax3 = axes[2]
    levels = list(bounds_result.crit_vals.index)
    lowers = [bounds_result.crit_vals.loc[lv, "lower"] for lv in levels]
    uppers = [bounds_result.crit_vals.loc[lv, "upper"] for lv in levels]
    x = np.arange(len(levels))
    width = 0.35
    ax3.bar(x - width / 2, lowers, width, color="#90CAF9", label="I(0) lower bound")
    ax3.bar(x + width / 2, uppers, width, color="#1976D2", label="I(1) upper bound")
    ax3.axhline(y=bounds_result.stat, color="#E91E63", linestyle="--",
                linewidth=2.5, label=f"F-stat = {bounds_result.stat:.3f}")
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(lv) for lv in levels])
    ax3.set_title("Bounds Test — F-stat vs Critical Values", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Significance level")
    ax3.set_ylabel("F-value")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3, axis="y")

    if has_eval:
        ax4 = axes[3]
        error_pct = ((merged_eval["Predicted_Low"] - merged_eval["Actual_Low"])
                     / merged_eval["Actual_Low"] * 100)
        colors = ["#4CAF50" if e >= 0 else "#F44336" for e in error_pct]
        ax4.bar(range(len(error_pct)), error_pct.values, color=colors, alpha=0.75, edgecolor="white")
        ax4.axhline(y=0, color="black", linewidth=0.8)
        ax4.axhline(y=error_pct.mean(), color="#FF9800", linestyle="--",
                    linewidth=2, label=f"Mean error: {error_pct.mean():.2f}%")
        ax4.set_xticks(range(len(error_pct)))
        ax4.set_xticklabels([d.strftime("%Y-%m-%d") for d in merged_eval.index], rotation=45)
        ax4.set_title("Error Harian Prediksi vs Aktual (%)", fontsize=13, fontweight="bold")
        ax4.set_ylabel("Error (%)")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_backtest(results_df, metrics, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    valid = results_df.dropna(subset=["predicted_low"]).copy()
    months_str = [m.strftime("%Y-%m") for m in valid["month"]]

    ax1 = axes[0]
    ax1.plot(months_str, valid["actual_low"] / 1e9, "o-", color="#4CAF50",
             linewidth=2, markersize=7, label="Aktual Monthly Low")
    ax1.plot(months_str, valid["predicted_low"] / 1e9, "s--", color="#E91E63",
             linewidth=2, markersize=7, label="Prediksi (expected_low)")
    ax1.fill_between(months_str, valid["ci_lower"] / 1e9, valid["ci_upper"] / 1e9,
                     color="#E91E63", alpha=0.15, label="CI (worst/best case)")
    ax1.set_title(f"Rolling Backtest — Actual vs Predicted Monthly Low (n={metrics['n']})",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Low (Miliar IDR)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    err = ((valid["predicted_low"] - valid["actual_low"]) / valid["actual_low"] * 100).values
    colors = ["#4CAF50" if e >= 0 else "#F44336" for e in err]
    ax2.bar(months_str, err, color=colors, alpha=0.75, edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axhline(y=err.mean(), color="#FF9800", linestyle="--",
                linewidth=2, label=f"Mean error: {err.mean():.2f}%")
    ax2.set_title(f"Error Bulanan (MAPE: {metrics['mape']:.2f}%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Error (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=45)

    ax3 = axes[2]
    in_ci_vals = valid["in_ci"].astype(int).values
    colors_ci = ["#4CAF50" if x else "#F44336" for x in in_ci_vals]
    ax3.bar(months_str, [1] * len(months_str), color=colors_ci, alpha=0.85, edgecolor="white")
    ax3.set_yticks([])
    ax3.set_title(f"CI Coverage: {metrics['ci_coverage']:.1f}% ({valid['in_ci'].sum()}/{len(valid)})  "
                  f"[hijau=in CI, merah=miss]", fontsize=13, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45)

    ax4 = axes[3]
    f_vals = valid["bounds_F"].values
    ax4.plot(months_str, f_vals, "o-", color="#1976D2", linewidth=2, markersize=7)
    ax4.axhline(y=4.0, color="#F44336", linestyle="--", linewidth=1.5,
                label="approx I(1) upper @5%")
    ax4.set_title(f"Bounds F-stat per Iterasi  ({metrics['coint_rate']:.0f}% iter cointegrated)",
                  fontsize=13, fontweight="bold")
    ax4.set_ylabel("F-statistic")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def get_forecast_params(cutoff_date=None):
    ref = cutoff_date if cutoff_date else TRAIN_END
    train_end_dt = datetime.strptime(ref, "%Y-%m-%d")
    last_train = pd.Timestamp(train_end_dt)
    target_month = (last_train.replace(day=1) + relativedelta(months=1))
    month_end = target_month + relativedelta(months=1) - pd.Timedelta(days=1)
    horizon = (month_end - last_train).days
    return horizon, target_month


def main():
    print("=" * 72)
    print("  BTC/IDR DAILY LOW — ARDL-ECM (Econometric Methodology)")
    print(f"  Mode          : {MODE}")
    print(f"  Training s/d  : {TRAIN_END}")
    win_str = f"{ROLLING_WINDOW_YEARS} tahun terakhir" if ROLLING_WINDOW_YEARS > 0 else "semua data (sejak 2014)"
    print(f"  Training window: {win_str}")
    print(f"  Log-transform : {'Ya (log Y & X)' if USE_LOG_TRANSFORM else 'Tidak (level)'}")
    if MODE == "forecast_and_evaluate":
        print(f"  Evaluasi s/d  : {EVAL_END}")
    elif MODE == "rolling_backtest":
        print(f"  Backtest      : {BACKTEST_MONTHS} bulan terakhir (walk-forward)")
    print("=" * 72)

    print("\n[1/8] Data Acquisition")
    daily = fetch_btc_daily(to_date=TRAIN_END)

    if MODE == "rolling_backtest":
        # ADF info sekali di awal pakai full data
        endog_info, exog_info = prepare_variables(apply_rolling_window(daily))
        check_stationarity(endog_info, exog_info)

        results = rolling_backtest_daily(daily, n_backtest_months=BACKTEST_MONTHS)
        metrics = evaluate_backtest(results)
        save_backtest_outputs(results, metrics, daily)

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
        return

    # --- Pipeline forecast_only / forecast_and_evaluate ---
    train = apply_rolling_window(daily)

    endog, exog = prepare_variables(train)
    check_stationarity(endog, exog)
    endog_lag, exog_orders = select_lag_order(endog, exog)
    ardl_fit = estimate_ardl(endog, exog, endog_lag, exog_orders)
    uecm_fit, bounds_result, cointegrated, cointegration_status = run_bounds_test(
        endog, exog, endog_lag, exog_orders
    )
    ecm_info = interpret_ecm(uecm_fit, cointegrated)

    horizon, target_month = get_forecast_params()
    print(f"\n      Target bulan forecast: {target_month.strftime('%Y-%m')}")
    print(f"      Horizon harian: {horizon} hari")

    exog_future = forecast_exog_var(exog, horizon)
    daily_forecast = forecast_daily(ardl_fit, endog, exog_future, horizon)
    monthly_summary = derive_monthly_low(daily_forecast, target_month)

    merged_eval = None
    eval_metrics = None
    if MODE == "forecast_and_evaluate":
        merged_eval, eval_metrics = evaluate_forecast(daily_forecast, target_month, EVAL_END)

    save_outputs(daily_forecast, monthly_summary, bounds_result, ecm_info,
                 cointegration_status, daily, target_month, merged_eval)

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Model         : ARDL(p={endog_lag}, orders={exog_orders}) via statsmodels")
    print(f"  Cointegration : {cointegration_status} (F={bounds_result.stat:.4f})")
    if ecm_info:
        print(f"  ECM λ         : {ecm_info['lambda']:.6f} (p={ecm_info['p']:.4f})")
    print(f"  Target month  : {monthly_summary['target_month']}")
    print(f"  Expected low  : Rp {monthly_summary['expected_low']:,.0f}")
    print(f"  Worst case    : Rp {monthly_summary['worst_case_low']:,.0f}")
    print(f"  Best case     : Rp {monthly_summary['best_case_low']:,.0f}")
    print(f"  Date of min   : {monthly_summary['date_of_min'].date()}")
    if eval_metrics:
        print(f"\n  Evaluasi vs Aktual:")
        print(f"    MAPE: {eval_metrics['mape']:.2f}%")
        print(f"    RMSE: Rp {eval_metrics['rmse']:,.0f}")
        print(f"    MAE : Rp {eval_metrics['mae']:,.0f}")
        print(f"    R²  : {eval_metrics['r2']:.4f}")
        print(f"    CI Coverage: {eval_metrics['ci_coverage']:.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
