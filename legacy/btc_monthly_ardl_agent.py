"""
BTC Monthly Low Prediction - Agent.md Specification
====================================================
Implementasi ARDL bulanan menggunakan statsmodels (sesuai spec Agent.md)
untuk perbandingan dengan versi harian (Ridge-based) di v3.

Flow:
1. Fetch data harian BTC/IDR dari Indodax API
2. Resample harian -> bulanan (low=min, high=max, close=last, open=first, volume=sum)
3. Normalisasi log-transform pada volume
4. Split train/test 80:20 sequential
5. Train ARDL via statsmodels.tsa.ardl.ARDL dengan lag selection AIC
6. Evaluate: MAPE, RMSE, MAE, R2 (target MAPE < 10%)
7. Forecast harga terendah bulan berikutnya
8. Output: CSV + PNG chart untuk compare dengan v3

Cara menjalankan:
    python btc_monthly_ardl_agent.py
"""

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# KONFIGURASI
# ============================================================================
TRAIN_TEST_SPLIT = 0.8
MAX_LAG_ENDOG = 6       # Max AR lags (bulan)
MAX_LAG_EXOG = 6        # Max DL lags (bulan)
IC = "aic"              # "aic" atau "bic"
TREND = "c"             # constant only
OUTPUT_DIR = "output"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "agent_monthly_predictions.csv")
OUTPUT_CHART = os.path.join(OUTPUT_DIR, "agent_monthly_chart.png")

# Indodax API
API_URL = "https://indodax.com/tradingview/history_v2"
SYMBOL = "BTCIDR"
TIMEFRAME = "1D"
FROM_TS = 1391191321  # 2014-02-01


# ============================================================================
# 1. DATA FETCH
# ============================================================================
def fetch_btc_daily():
    """Fetch data harian BTC/IDR dari Indodax API sampai hari ini."""
    # Fetch sampai akhir hari ini
    now = datetime.now()
    to_ts = int(now.timestamp())

    params = {
        "from": FROM_TS,
        "to": to_ts,
        "symbol": SYMBOL,
        "tf": TIMEFRAME,
    }

    print(f"Fetching data dari Indodax API (sampai {now.strftime('%Y-%m-%d')})...")
    response = requests.get(API_URL, params=params)

    if response.status_code != 200:
        raise RuntimeError(f"Gagal fetch data: HTTP {response.status_code}")

    data = response.json()
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["Time"], unit="s")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    # Cleaning
    df = df.drop_duplicates(subset="Time", keep="last").reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill()
        df[col] = df[col].interpolate(method="linear")

    print(f"Data harian bersih: {len(df)} records ({df['Time'].min().date()} - {df['Time'].max().date()})")
    return df


# ============================================================================
# 2. RESAMPLE HARIAN -> BULANAN
# ============================================================================
def resample_to_monthly(daily_df):
    """
    Agregasi data harian ke bulanan.

    - monthly_open:  first harga open di bulan tsb
    - monthly_high:  max harga high di bulan tsb
    - monthly_low:   min harga low di bulan tsb (TARGET)
    - monthly_close: last harga close di bulan tsb
    - total_volume:  sum volume di bulan tsb
    - log_volume:    log1p(total_volume) untuk normalisasi
    """
    df = daily_df.set_index("Time").sort_index()

    monthly = df.resample("M").agg(
        monthly_open=("Open", "first"),
        monthly_high=("High", "max"),
        monthly_low=("Low", "min"),
        monthly_close=("Close", "last"),
        total_volume=("Volume", "sum"),
    )

    # Drop bulan terakhir jika partial (kurang dari 20 hari data di bulan itu)
    last_month = monthly.index[-1]
    days_in_last_month = len(df[df.index.to_period("M") == last_month.to_period("M")])
    if days_in_last_month < 20:
        print(f"Drop bulan terakhir ({last_month.strftime('%Y-%m')}) karena partial: {days_in_last_month} hari saja")
        monthly = monthly.iloc[:-1]

    monthly["log_volume"] = np.log1p(monthly["total_volume"])
    monthly = monthly.dropna()

    # Set frequency agar statsmodels ARDL tidak protes
    monthly = monthly.asfreq("M")

    print(f"Data bulanan: {len(monthly)} records ({monthly.index.min().strftime('%Y-%m')} - {monthly.index.max().strftime('%Y-%m')})")
    return monthly


# ============================================================================
# 3. STATIONARITY CHECK
# ============================================================================
def check_stationarity(monthly_df):
    """ADF test untuk variabel-variabel bulanan."""
    print("\n" + "=" * 70)
    print("  ADF Stationarity Test (Level)")
    print("=" * 70)
    print(f"  {'Variabel':<20} {'ADF Stat':>12} {'p-value':>12} {'Status':<20}")
    print("  " + "-" * 68)

    columns = ["monthly_low", "monthly_close", "monthly_high", "log_volume"]
    results = {}

    for col in columns:
        series = monthly_df[col].dropna()
        try:
            adf_stat, p_value = adfuller(series, autolag="AIC")[:2]
            is_stationary = p_value < 0.05
            status = "Stasioner I(0)" if is_stationary else "Non-stasioner"
            results[col] = (adf_stat, p_value, is_stationary)
            print(f"  {col:<20} {adf_stat:>12.4f} {p_value:>12.4f} {status:<20}")
        except Exception as e:
            print(f"  {col:<20} ADF gagal: {e}")

    # First difference untuk non-stasioner
    non_stat = [c for c, r in results.items() if not r[2]]
    if non_stat:
        print("\n  First Difference Test:")
        for col in non_stat:
            diff = monthly_df[col].diff().dropna()
            try:
                adf_stat, p_value = adfuller(diff, autolag="AIC")[:2]
                status = "I(1) OK" if p_value < 0.05 else "I(2) WARNING"
                print(f"  d({col}): ADF={adf_stat:.4f}, p={p_value:.4f} -> {status}")
            except Exception as e:
                print(f"  d({col}): ADF gagal: {e}")

    print("  (Catatan: ARDL valid untuk campuran I(0) dan I(1))")
    return results


# ============================================================================
# 4. LAG SELECTION (AIC)
# ============================================================================
def select_lag_order(train_df):
    """
    Gunakan ardl_select_order() dari statsmodels untuk pilih lag optimal.

    endog: monthly_low
    exog:  monthly_close, monthly_high, log_volume
    """
    print("\n" + "=" * 70)
    print(f"  Lag Selection (statsmodels ARDL, ic={IC})")
    print("=" * 70)

    endog = train_df["monthly_low"]
    exog = train_df[["monthly_close", "monthly_high", "log_volume"]]

    print(f"  Max lag endog: {MAX_LAG_ENDOG}, Max order exog: {MAX_LAG_EXOG}")
    print("  Mencari kombinasi terbaik...")

    sel = ardl_select_order(
        endog=endog,
        maxlag=MAX_LAG_ENDOG,
        exog=exog,
        maxorder=MAX_LAG_EXOG,
        ic=IC,
        trend=TREND,
    )

    selected_model = sel.model
    ardl_order = selected_model.ardl_order
    endog_lag = ardl_order[0]
    exog_orders = ardl_order[1:]

    print(f"  Best endog lag: {endog_lag}")
    print(f"  Best exog orders: monthly_close={exog_orders[0]}, "
          f"monthly_high={exog_orders[1]}, log_volume={exog_orders[2]}")
    print(f"  {IC.upper()}: {getattr(selected_model.fit(), IC):.4f}")

    return endog_lag, exog_orders


# ============================================================================
# 5. TRAIN ARDL
# ============================================================================
def train_ardl(train_df, endog_lag, exog_orders):
    """Train ARDL model pada training set."""
    print("\n" + "=" * 70)
    print("  Training ARDL Model")
    print("=" * 70)

    endog = train_df["monthly_low"]
    exog = train_df[["monthly_close", "monthly_high", "log_volume"]]

    model = ARDL(
        endog=endog,
        lags=endog_lag,
        exog=exog,
        order={
            "monthly_close": exog_orders[0],
            "monthly_high": exog_orders[1],
            "log_volume": exog_orders[2],
        },
        trend=TREND,
    )
    fit = model.fit()

    # R2 dari in-sample fitted values
    fitted = fit.fittedvalues
    y_true = endog.loc[fitted.index]
    r2_train = r2_score(y_true, fitted)

    print(f"  Training samples: {len(train_df)}")
    print(f"  AIC: {fit.aic:.4f}")
    print(f"  BIC: {fit.bic:.4f}")
    print(f"  R-squared (training): {r2_train:.4f}")

    return fit


# ============================================================================
# 6. EVALUATE ON TEST SET
# ============================================================================
def evaluate_model(fit, monthly_df, split_idx):
    """
    Out-of-sample evaluation menggunakan fit.predict() dengan exog_oos.
    """
    print("\n" + "=" * 70)
    print("  Out-of-Sample Evaluation")
    print("=" * 70)

    test_df = monthly_df.iloc[split_idx:]
    test_exog = test_df[["monthly_close", "monthly_high", "log_volume"]]

    # Predict: start=split_idx (posisi integer), end=len-1
    # dynamic=False berarti one-step-ahead (pakai aktual sebagai lag, bukan prediksi)
    predictions = fit.predict(
        start=split_idx,
        end=len(monthly_df) - 1,
        exog_oos=test_exog,
        dynamic=False,
    )

    # Align: predictions punya index dari monthly_df
    aligned = pd.DataFrame({
        "actual_low": test_df["monthly_low"].values,
        "predicted_low": predictions.values[-len(test_df):],
    }, index=test_df.index)

    # Hitung metrics
    actual = aligned["actual_low"].values
    pred = aligned["predicted_low"].values

    metrics = {
        "mape": mean_absolute_percentage_error(actual, pred) * 100,
        "rmse": np.sqrt(mean_squared_error(actual, pred)),
        "mae": np.mean(np.abs(actual - pred)),
        "r2": r2_score(actual, pred),
    }

    aligned["error_rp"] = aligned["predicted_low"] - aligned["actual_low"]
    aligned["error_pct"] = (aligned["error_rp"].abs() / aligned["actual_low"]) * 100

    print(f"  Test samples: {len(test_df)}")
    print(f"  Periode: {test_df.index.min().strftime('%Y-%m')} - {test_df.index.max().strftime('%Y-%m')}")
    print(f"  MAPE: {metrics['mape']:>8.2f}%  {'[TARGET TERCAPAI]' if metrics['mape'] < 10 else '[Target MAPE < 10% belum tercapai]'}")
    print(f"  RMSE: Rp {metrics['rmse']:>14,.0f}")
    print(f"  MAE:  Rp {metrics['mae']:>14,.0f}")
    print(f"  R2:   {metrics['r2']:>8.4f}")

    return aligned, metrics


# ============================================================================
# 7. FORECAST NEXT MONTH
# ============================================================================
def forecast_next_month(fit, monthly_df):
    """
    Prediksi monthly_low untuk bulan berikutnya.
    Proxy exog untuk bulan depan: nilai terakhir (persistence assumption).
    """
    print("\n" + "=" * 70)
    print("  Forecast Bulan Berikutnya")
    print("=" * 70)

    last_row = monthly_df.iloc[-1]
    last_month = monthly_df.index[-1]
    next_month = last_month + pd.offsets.MonthEnd(1)

    # Proxy exog untuk 1 step ke depan
    proxy_exog = pd.DataFrame({
        "monthly_close": [last_row["monthly_close"]],
        "monthly_high": [last_row["monthly_high"]],
        "log_volume": [last_row["log_volume"]],
    }, index=pd.DatetimeIndex([next_month], freq="M"))

    forecast = fit.forecast(steps=1, exog=proxy_exog)
    forecast_value = float(forecast.iloc[0])

    print(f"  Bulan target: {next_month.strftime('%Y-%m')}")
    print(f"  Prediksi harga terendah: Rp {forecast_value:,.0f}")
    print(f"  (proxy exog pakai nilai bulan {last_month.strftime('%Y-%m')})")

    return next_month, forecast_value


# ============================================================================
# 8. VISUALISASI
# ============================================================================
def plot_results(monthly_df, aligned, split_idx, forecast_month, forecast_value, metrics):
    """3-panel chart: historis+prediksi, residuals, error distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # Panel 1: Historis + Prediksi + Forecast
    ax1 = axes[0]
    ax1.plot(monthly_df.index, monthly_df["monthly_low"] / 1e9,
             label="Aktual (historis)", color="#2196F3", linewidth=1.8)
    ax1.plot(aligned.index, aligned["predicted_low"] / 1e9,
             label="Prediksi (test)", color="#E91E63", linewidth=2, linestyle="--")

    # Split line
    split_date = monthly_df.index[split_idx]
    ax1.axvline(x=split_date, color="gray", linestyle=":", linewidth=1.5, label="Train/Test Split")

    # Forecast bulan depan
    ax1.scatter([forecast_month], [forecast_value / 1e9],
                color="#FF9800", s=120, zorder=5,
                label=f"Forecast {forecast_month.strftime('%Y-%m')}", marker="*")

    ax1.set_title(f"BTC/IDR Monthly Low - ARDL (statsmodels) | MAPE: {metrics['mape']:.2f}%",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Bulan")
    ax1.set_ylabel("Harga Low (Miliar IDR)")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Panel 2: Actual vs Predicted (test only)
    ax2 = axes[1]
    ax2.plot(aligned.index, aligned["actual_low"] / 1e9,
             label="Aktual", color="#2196F3", linewidth=2, marker="o", markersize=6)
    ax2.plot(aligned.index, aligned["predicted_low"] / 1e9,
             label="Prediksi", color="#E91E63", linewidth=2, marker="s", markersize=6, linestyle="--")
    ax2.fill_between(aligned.index,
                     aligned["actual_low"] / 1e9,
                     aligned["predicted_low"] / 1e9,
                     alpha=0.2, color="gray")
    ax2.set_title("Test Period: Aktual vs Prediksi", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Bulan")
    ax2.set_ylabel("Harga Low (Miliar IDR)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # Panel 3: Error distribution
    ax3 = axes[2]
    error_signed = ((aligned["predicted_low"] - aligned["actual_low"])
                    / aligned["actual_low"] * 100)
    colors = ["#4CAF50" if e >= 0 else "#F44336" for e in error_signed]
    ax3.bar(range(len(error_signed)), error_signed.values, color=colors, alpha=0.75, edgecolor="white")
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.axhline(y=error_signed.mean(), color="#FF9800", linestyle="--",
                linewidth=2, label=f"Mean: {error_signed.mean():.2f}%")
    ax3.set_xticks(range(len(error_signed)))
    ax3.set_xticklabels([d.strftime("%Y-%m") for d in aligned.index], rotation=45)
    ax3.set_title(f"Error per Bulan (%) | Target MAPE < 10%", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Bulan")
    ax3.set_ylabel("Error (%)")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nChart disimpan: {OUTPUT_CHART}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  BTC MONTHLY LOW PREDICTION - Agent.md Specification")
    print("  ARDL via statsmodels.tsa.ardl")
    print("=" * 70)

    # 1. Fetch
    daily = fetch_btc_daily()

    # 2. Resample
    monthly = resample_to_monthly(daily)

    if len(monthly) < 30:
        raise RuntimeError(f"Data bulanan terlalu sedikit: {len(monthly)}")

    # 3. ADF
    check_stationarity(monthly)

    # 4. Split
    split_idx = int(len(monthly) * TRAIN_TEST_SPLIT)
    train_df = monthly.iloc[:split_idx]
    test_df = monthly.iloc[split_idx:]

    print(f"\nSplit: train={len(train_df)} ({train_df.index.min().strftime('%Y-%m')} - {train_df.index.max().strftime('%Y-%m')}), "
          f"test={len(test_df)} ({test_df.index.min().strftime('%Y-%m')} - {test_df.index.max().strftime('%Y-%m')})")

    # 5. Lag selection (pada training data saja, hindari leakage)
    endog_lag, exog_orders = select_lag_order(train_df)

    # 6. Train
    fit = train_ardl(train_df, endog_lag, exog_orders)

    # Print ringkasan model (opsional, full summary bisa sangat panjang)
    print("\n  Parameter estimates (top 10):")
    params = fit.params
    for name, val in list(params.items())[:10]:
        print(f"    {name:<35} {val:>14,.4f}")

    # 7. Evaluate
    aligned, metrics = evaluate_model(fit, monthly, split_idx)

    # 8. Forecast next month
    forecast_month, forecast_value = forecast_next_month(fit, monthly)

    # 9. Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_df = aligned.copy()
    csv_df.insert(0, "month_year", csv_df.index.strftime("%Y-%m"))
    csv_df = csv_df.reset_index(drop=True)
    csv_df["actual_low"] = csv_df["actual_low"].apply(lambda x: f"Rp {x:,.0f}")
    csv_df["predicted_low"] = csv_df["predicted_low"].apply(lambda x: f"Rp {x:,.0f}")
    csv_df["error_rp"] = csv_df["error_rp"].apply(lambda x: f"Rp {x:,.0f}")
    csv_df["error_pct"] = csv_df["error_pct"].apply(lambda x: f"{x:.2f}%")

    # Tambah baris forecast
    forecast_row = pd.DataFrame([{
        "month_year": forecast_month.strftime("%Y-%m") + " (FORECAST)",
        "actual_low": "-",
        "predicted_low": f"Rp {forecast_value:,.0f}",
        "error_rp": "-",
        "error_pct": "-",
    }])
    csv_df = pd.concat([csv_df, forecast_row], ignore_index=True)

    csv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV disimpan: {OUTPUT_CSV}")

    # 10. Visualisasi
    plot_results(monthly, aligned, split_idx, forecast_month, forecast_value, metrics)

    # 11. Summary & comparison
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model: ARDL(p={endog_lag}, q={exog_orders}) via statsmodels")
    print(f"  Training: {len(train_df)} bulan ({train_df.index.min().strftime('%Y-%m')} - {train_df.index.max().strftime('%Y-%m')})")
    print(f"  Testing:  {len(test_df)} bulan ({test_df.index.min().strftime('%Y-%m')} - {test_df.index.max().strftime('%Y-%m')})")
    print(f"\n  Metrics (out-of-sample):")
    print(f"    MAPE : {metrics['mape']:.2f}%")
    print(f"    RMSE : Rp {metrics['rmse']:,.0f}")
    print(f"    MAE  : Rp {metrics['mae']:,.0f}")
    print(f"    R2   : {metrics['r2']:.4f}")
    print(f"\n  Forecast {forecast_month.strftime('%Y-%m')}: Rp {forecast_value:,.0f}")
    print(f"  (Rekomendasi harga beli DCA bulan depan)")

    print("\n" + "=" * 70)
    print("  PERBANDINGAN vs v3 (harian recursive):")
    print(f"  - v3 MAPE    : ~3-5% (prediction_ardl_v3_eval_jan2026.csv)")
    print(f"  - Agent MAPE : {metrics['mape']:.2f}% (monthly, statsmodels)")
    print("  Catatan: v3 evaluate 31 hari, versi ini evaluate ~{} bulan".format(len(test_df)))
    print("=" * 70)


if __name__ == "__main__":
    main()
