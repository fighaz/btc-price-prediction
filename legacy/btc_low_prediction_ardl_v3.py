"""
BTC/IDR Low Price Prediction - Production-Ready Forecasting
Menggunakan Metode ARDL (Autoregressive Distributed Lags) v3

Production workflow:
1. AWAL BULAN  (forecast_only): Train model dengan data historis → prediksi bulan depan
2. AKHIR BULAN (forecast_and_evaluate): Fetch data aktual → bandingkan dengan prediksi

Fitur baru vs v2:
- Recursive forecasting: prediksi N hari ke depan TANPA data aktual
- Dua mode operasi: forecast_only dan forecast_and_evaluate
- Confidence interval yang melebar seiring horizon
- Dynamic data fetch (to_date parameter)
- Optimasi: AIC grid search lebih halus, alpha range diperluas
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# KONFIGURASI
# ============================================================================

MODE = "forecast_and_evaluate"   # "forecast_only" atau "forecast_and_evaluate"
TRAIN_DATA_END = "2025-12-31"    # Training data berakhir di sini
FORECAST_DAYS = 31               # Prediksi 31 hari ke depan (Januari 2026)
EVAL_DATA_END = "2026-01-31"     # Untuk evaluasi: fetch aktual sampai sini
MIN_LAG = 2                      # Minimum lag offset (anti-overfitting)

BASE_DIR = '/home/fighaz/Documents/btc-price-prediction'


# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================

def fetch_btc_data(to_date=None):
    """
    Fetch data BTC/IDR dari Indodax API

    Parameters:
    - to_date: string "YYYY-MM-DD" atau None
      None = default 2025-12-31 (training data)
      String = fetch sampai tanggal tersebut
    """
    url = "https://indodax.com/tradingview/history_v2"

    from_ts = 1391191321  # 2014-02-01

    if to_date is None:
        to_ts = 1767200399  # 2025-12-31
        label = "2025-12-31"
    else:
        dt = datetime.strptime(to_date, "%Y-%m-%d")
        dt = dt.replace(hour=23, minute=59, second=59)
        to_ts = int(dt.timestamp())
        label = to_date

    params = {
        "from": from_ts,
        "to": to_ts,
        "symbol": "BTCIDR",
        "tf": "1D"
    }

    print(f"Fetching data dari Indodax API (sampai {label})...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Gagal fetch data: {response.status_code}")
        return None

    data = response.json()
    df = pd.DataFrame(data)

    df["Time"] = pd.to_datetime(df["Time"], unit="s")

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    print(f"Data mentah: {len(df)} records")
    print(f"Rentang data: {df['Time'].min()} - {df['Time'].max()}")

    # --- Data Cleaning ---
    before_dedup = len(df)
    df = df.drop_duplicates(subset='Time', keep='last').reset_index(drop=True)
    if before_dedup > len(df):
        print(f"Duplikat dihapus: {before_dedup - len(df)} baris")

    df = df.sort_values('Time').reset_index(drop=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill()
        df[col] = df[col].interpolate(method='linear')

    for col in ['Open', 'High', 'Low', 'Close']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        anomalies = ((df[col] < lower) | (df[col] > upper)).sum()
        if anomalies > 0:
            print(f"Anomali di {col}: {anomalies} titik data (tidak dihapus)")

    print(f"Data bersih: {len(df)} records")
    return df


# ============================================================================
# 2. STATISTICAL TESTS
# ============================================================================

def run_adf_test(df):
    """ADF Test untuk stasioneritas"""
    print("\nAugmented Dickey-Fuller (ADF) Stationarity Test:")
    print("-" * 65)
    print(f"  {'Variabel':<12} {'ADF Stat':>10} {'p-value':>10} {'Status':<15}")
    print("-" * 65)

    columns_to_test = ['Low', 'Open', 'High', 'Close', 'Volume']
    results = {}

    for col in columns_to_test:
        result = adfuller(df[col].dropna(), autolag='AIC')
        adf_stat, p_value = result[0], result[1]
        is_stationary = p_value < 0.05
        status = "Stasioner I(0)" if is_stationary else "Non-stasioner"
        results[col] = {'adf_statistic': adf_stat, 'p_value': p_value, 'stationary': is_stationary}
        print(f"  {col:<12} {adf_stat:>10.4f} {p_value:>10.4f} {status:<15}")

    non_stationary = [col for col, r in results.items() if not r['stationary']]
    if non_stationary:
        print("\n  First Difference Test (variabel non-stasioner):")
        for col in non_stationary:
            diff_result = adfuller(df[col].diff().dropna(), autolag='AIC')
            diff_p = diff_result[1]
            diff_status = "I(1) - OK" if diff_p < 0.05 else "I(2) - WARNING"
            print(f"  d({col}): ADF={diff_result[0]:.4f}, p={diff_p:.4f} -> {diff_status}")

    print("-" * 65)
    print("  Catatan: ARDL valid untuk campuran variabel I(0) dan I(1)")
    return results


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def prepare_ardl_features(df, p_lags=7, q_lags=7, min_lag=2):
    """
    Feature engineering untuk ARDL model (anti-overfitting)

    Anti-overfitting measures:
    - Minimum lag = 2 (bukan 1) agar model tidak menyalin harga kemarin
    - MA & EMA dari Close (independen), bukan Low (target)
    - Volatilitas dari spread (High-Low), bukan Low
    - Percentage return, bukan absolute diff
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    df['DayIndex'] = range(len(df))

    # AR lags (Low)
    for i in range(min_lag, min_lag + p_lags):
        df[f'Low_lag{i}'] = df['Low'].shift(i)

    # DL lags (Open, High, Close, Volume)
    for i in range(min_lag, min_lag + q_lags):
        df[f'Open_lag{i}'] = df['Open'].shift(i)
    for i in range(min_lag, min_lag + q_lags):
        df[f'High_lag{i}'] = df['High'].shift(i)
    for i in range(min_lag, min_lag + q_lags):
        df[f'Close_lag{i}'] = df['Close'].shift(i)

    df['Log_Volume'] = np.log1p(df['Volume'])
    for i in range(min_lag, min_lag + q_lags):
        df[f'LogVolume_lag{i}'] = df['Log_Volume'].shift(i)

    # Momentum (percentage returns)
    df['Low_pct'] = df['Low'].pct_change().shift(min_lag)
    df['Open_pct'] = df['Open'].pct_change().shift(min_lag)
    df['High_pct'] = df['High'].pct_change().shift(min_lag)
    df['Close_pct'] = df['Close'].pct_change().shift(min_lag)

    # MA & EMA dari Close
    df['MA7_Close'] = df['Close'].rolling(window=7).mean().shift(min_lag)
    df['MA14_Close'] = df['Close'].rolling(window=14).mean().shift(min_lag)
    df['MA30_Close'] = df['Close'].rolling(window=30).mean().shift(min_lag)
    df['EMA7_Close'] = df['Close'].ewm(span=7, adjust=False).mean().shift(min_lag)
    df['EMA14_Close'] = df['Close'].ewm(span=14, adjust=False).mean().shift(min_lag)

    # Volatilitas dari spread
    df['HL_Spread'] = df['High'] - df['Low']
    df['Volatility_7d'] = df['HL_Spread'].rolling(window=7).std().shift(min_lag)
    df['Volatility_14d'] = df['HL_Spread'].rolling(window=14).std().shift(min_lag)

    for i in range(min_lag, min_lag + 3):
        df[f'HL_Spread_lag{i}'] = df['HL_Spread'].shift(i)

    df['OC_Spread'] = df['Close'] - df['Open']
    for i in range(min_lag, min_lag + 3):
        df[f'OC_Spread_lag{i}'] = df['OC_Spread'].shift(i)

    df = df.dropna().reset_index(drop=True)
    return df


def get_ardl_features(p_lags=7, q_lags=7, min_lag=2):
    """Get list of feature names"""
    features = []

    for i in range(min_lag, min_lag + p_lags):
        features.append(f'Low_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'Open_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'High_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'Close_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'LogVolume_lag{i}')

    features.extend(['Low_pct', 'Open_pct', 'High_pct', 'Close_pct'])
    features.extend(['MA7_Close', 'MA14_Close', 'MA30_Close', 'EMA7_Close', 'EMA14_Close'])
    features.extend(['Volatility_7d', 'Volatility_14d'])

    for i in range(min_lag, min_lag + 3):
        features.append(f'HL_Spread_lag{i}')
    for i in range(min_lag, min_lag + 3):
        features.append(f'OC_Spread_lag{i}')

    return features


# ============================================================================
# 4. PARAMETER OPTIMIZATION (AIC)
# ============================================================================

def select_optimal_lags(df, max_p=10, max_q=10, min_lag=2):
    """
    Select optimal lag order menggunakan AIC
    Grid search step=1 (lebih halus dari v2 yang step=2)
    """
    print("\nMencari optimal lag order menggunakan AIC...")
    print(f"  (grid search step=1, min_lag={min_lag})")

    best_aic = np.inf
    best_p = 1
    best_q = 1
    results = []

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                temp_df = prepare_ardl_features(df.copy(), p_lags=p, q_lags=q, min_lag=min_lag)
                if len(temp_df) < 100:
                    continue

                train_size = int(len(temp_df) * 0.8)
                train_df = temp_df.iloc[:train_size]

                features = get_ardl_features(p, q, min_lag=min_lag)
                available_features = [f for f in features if f in train_df.columns]

                X = train_df[available_features].values
                y = train_df['Low'].values

                model = LinearRegression()
                model.fit(X, y)

                y_pred = model.predict(X)
                n = len(y)
                k = X.shape[1] + 1
                rss = np.sum((y - y_pred) ** 2)
                aic = n * np.log(rss / n) + 2 * k

                results.append({'p': p, 'q': q, 'aic': aic})

                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    best_q = q
            except Exception:
                continue

    print(f"  Optimal lags: p={best_p} (AR), q={best_q} (DL)")
    print(f"  Actual lag range: {min_lag} - {min_lag + max(best_p, best_q) - 1}")
    print(f"  AIC: {best_aic:.2f}")

    if results:
        sorted_results = sorted(results, key=lambda x: x['aic'])[:5]
        print("\n  Top 5 kombinasi lag (AIC terendah):")
        for i, r in enumerate(sorted_results, 1):
            marker = " <-- terpilih" if r['p'] == best_p and r['q'] == best_q else ""
            print(f"    {i}. p={r['p']}, q={r['q']}, AIC={r['aic']:.2f}{marker}")

    return best_p, best_q


# ============================================================================
# 5. ARDL MODEL
# ============================================================================

class ARDLModel:
    """ARDL Model dengan Ridge regularization"""

    def __init__(self, p_lags=7, q_lags=7, regularization='ridge', alpha=1.0):
        self.p_lags = p_lags
        self.q_lags = q_lags
        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.residual_std = None  # Untuk confidence interval

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)

        if hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)

        # Hitung residual std untuk confidence interval
        y_pred = self.model.predict(X_scaled)
        self.residual_std = np.std(y - y_pred)

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_top_features(self, n=10):
        if self.feature_importance is None or self.feature_names is None:
            return []
        indices = np.argsort(self.feature_importance)[::-1][:n]
        return [(self.feature_names[i], self.feature_importance[i]) for i in indices]


# ============================================================================
# 6. TRAINING
# ============================================================================

def train_ardl_model(df, p_lags=7, q_lags=7, min_lag=2):
    """
    Training ARDL model pada SELURUH data (untuk production forecasting)
    """
    df_features = prepare_ardl_features(df.copy(), p_lags, q_lags, min_lag=min_lag)

    features = get_ardl_features(p_lags, q_lags, min_lag=min_lag)
    available_features = [f for f in features if f in df_features.columns]

    X_train = df_features[available_features].values
    y_train = df_features['Low'].values

    print(f"\nTraining ARDL Model (seluruh data)...")
    print(f"  Data: {len(df_features)} samples ({df_features['Time'].min().date()} - {df_features['Time'].max().date()})")
    print(f"  p-lags (AR): {p_lags}, q-lags (DL): {q_lags}, min_lag: {min_lag}")
    print(f"  Total features: {len(available_features)}")

    # CV untuk alpha (range diperluas dari v2)
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha = 1.0
    best_cv_score = -np.inf

    for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            temp_model = Ridge(alpha=alpha)
            scaler = StandardScaler()
            X_cv_train_scaled = scaler.fit_transform(X_cv_train)
            X_cv_val_scaled = scaler.transform(X_cv_val)

            temp_model.fit(X_cv_train_scaled, y_cv_train)
            score = temp_model.score(X_cv_val_scaled, y_cv_val)
            cv_scores.append(score)

        avg_score = np.mean(cv_scores)
        if avg_score > best_cv_score:
            best_cv_score = avg_score
            best_alpha = alpha

    print(f"  Best alpha (via CV): {best_alpha}")

    # Train final model
    ardl_model = ARDLModel(p_lags=p_lags, q_lags=q_lags, regularization='ridge', alpha=best_alpha)
    ardl_model.fit(X_train, y_train, available_features)

    train_r2 = r2_score(y_train, ardl_model.predict(X_train))
    print(f"  R-squared (training): {train_r2:.4f}")
    print(f"  Residual std: Rp {ardl_model.residual_std:,.0f}")

    print("\n  Top 10 Most Important Features:")
    for feat, importance in ardl_model.get_top_features(10):
        print(f"    - {feat}: {importance:.4f}")

    return ardl_model, df_features, available_features


# ============================================================================
# 7. RECURSIVE FORECASTING (INTI PRODUCTION)
# ============================================================================

def compute_single_row_features(history_df, p_lags, q_lags, min_lag, feature_names):
    """
    Hitung feature vector untuk SATU baris terakhir dari history buffer.
    Mirip logika prepare_ardl_features() tapi untuk single row.

    Parameters:
    - history_df: DataFrame dengan kolom [Time, Open, High, Low, Close, Volume]
                  Harus cukup panjang untuk semua lag + rolling window
    - p_lags, q_lags, min_lag: parameter ARDL
    - feature_names: list nama fitur yang dibutuhkan model

    Returns:
    - numpy array (1, n_features) siap untuk model.predict()
    """
    df = history_df.copy()
    df = df.reset_index(drop=True)
    last_idx = len(df) - 1

    feature_values = {}

    # AR lags (Low)
    for i in range(min_lag, min_lag + p_lags):
        col = f'Low_lag{i}'
        if col in feature_names:
            feature_values[col] = df['Low'].iloc[last_idx - i]

    # DL lags
    for var_name, col_name in [('Open', 'Open'), ('High', 'High'), ('Close', 'Close')]:
        for i in range(min_lag, min_lag + q_lags):
            fname = f'{var_name}_lag{i}'
            if fname in feature_names:
                feature_values[fname] = df[col_name].iloc[last_idx - i]

    # LogVolume lags
    log_vol = np.log1p(df['Volume'].values)
    for i in range(min_lag, min_lag + q_lags):
        fname = f'LogVolume_lag{i}'
        if fname in feature_names:
            feature_values[fname] = log_vol[last_idx - i]

    # Percentage returns (shifted by min_lag)
    for var_name in ['Low', 'Open', 'High', 'Close']:
        fname = f'{var_name}_pct'
        if fname in feature_names:
            idx = last_idx - min_lag
            if idx > 0:
                prev = df[var_name].iloc[idx - 1]
                curr = df[var_name].iloc[idx]
                feature_values[fname] = (curr - prev) / prev if prev != 0 else 0
            else:
                feature_values[fname] = 0

    # Moving averages dari Close (shifted by min_lag)
    close_vals = df['Close'].values
    for window, fname in [(7, 'MA7_Close'), (14, 'MA14_Close'), (30, 'MA30_Close')]:
        if fname in feature_names:
            end_idx = last_idx - min_lag + 1
            start_idx = max(0, end_idx - window)
            if end_idx > start_idx:
                feature_values[fname] = np.mean(close_vals[start_idx:end_idx])
            else:
                feature_values[fname] = close_vals[last_idx - min_lag]

    # EMA dari Close (shifted by min_lag)
    for span, fname in [(7, 'EMA7_Close'), (14, 'EMA14_Close')]:
        if fname in feature_names:
            end_idx = last_idx - min_lag + 1
            series = pd.Series(close_vals[:end_idx])
            ema_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
            feature_values[fname] = ema_val

    # Volatility dari HL spread (shifted by min_lag)
    hl_spread = df['High'].values - df['Low'].values
    for window, fname in [(7, 'Volatility_7d'), (14, 'Volatility_14d')]:
        if fname in feature_names:
            end_idx = last_idx - min_lag + 1
            start_idx = max(0, end_idx - window)
            if end_idx - start_idx > 1:
                feature_values[fname] = np.std(hl_spread[start_idx:end_idx], ddof=1)
            else:
                feature_values[fname] = 0

    # HL_Spread lags
    for i in range(min_lag, min_lag + 3):
        fname = f'HL_Spread_lag{i}'
        if fname in feature_names:
            feature_values[fname] = hl_spread[last_idx - i]

    # OC_Spread lags
    oc_spread = df['Close'].values - df['Open'].values
    for i in range(min_lag, min_lag + 3):
        fname = f'OC_Spread_lag{i}'
        if fname in feature_names:
            feature_values[fname] = oc_spread[last_idx - i]

    # Susun array sesuai urutan feature_names
    row = np.array([feature_values.get(f, 0) for f in feature_names]).reshape(1, -1)
    return row


def forecast_future(ardl_model, df, feature_names, n_days=31,
                    p_lags=9, q_lags=9, min_lag=2):
    """
    Recursive forecasting: prediksi N hari ke depan TANPA data aktual.

    Algoritma:
    1. Gunakan data historis terakhir sebagai seed
    2. Prediksi Low_t → isi proxy OHLCV → prediksi Low_{t+1} → repeat
    3. Confidence interval melebar seiring horizon

    Parameters:
    - ardl_model: trained ARDLModel
    - df: DataFrame historis lengkap (sudah cleaned)
    - feature_names: list fitur yang digunakan model
    - n_days: jumlah hari yang diprediksi
    - p_lags, q_lags, min_lag: parameter ARDL

    Returns:
    - DataFrame [Time, Predicted_Low, Confidence_Lower, Confidence_Upper]
    """
    print(f"\n{'='*50}")
    print(f"  RECURSIVE FORECAST: {n_days} hari ke depan")
    print(f"{'='*50}")

    # Hitung rasio untuk proxy OHLCV dari 30 hari terakhir data asli
    last_30 = df.tail(30)
    high_low_ratio = (last_30['High'] / last_30['Low']).median()
    close_low_ratio = (last_30['Close'] / last_30['Low']).median()
    median_volume = last_30['Volume'].median()

    print(f"  Proxy ratios (dari 30 hari terakhir data asli):")
    print(f"    High/Low ratio : {high_low_ratio:.6f}")
    print(f"    Close/Low ratio: {close_low_ratio:.6f}")
    print(f"    Median Volume  : {median_volume:,.0f}")

    # Siapkan history buffer (cukup untuk semua lag + rolling window)
    buffer_size = min_lag + max(p_lags, q_lags) + 35
    history = df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(buffer_size).copy()
    history = history.reset_index(drop=True)

    last_date = history['Time'].iloc[-1]
    residual_std = ardl_model.residual_std

    predictions = []

    print(f"\n  Forecasting dari {last_date.date()} + 1 hari...")
    print(f"  Residual std (base): Rp {residual_std:,.0f}")

    for step in range(1, n_days + 1):
        forecast_date = last_date + timedelta(days=step)

        # Hitung features dari history buffer
        feature_row = compute_single_row_features(
            history, p_lags, q_lags, min_lag, feature_names
        )

        # Prediksi Low
        pred_low = ardl_model.predict(feature_row)[0]

        # Confidence interval (melebar seiring step)
        ci_width = residual_std * np.sqrt(step) * 1.96
        ci_lower = pred_low - ci_width
        ci_upper = pred_low + ci_width

        predictions.append({
            'Time': forecast_date,
            'Predicted_Low': pred_low,
            'Confidence_Lower': ci_lower,
            'Confidence_Upper': ci_upper,
            'Step': step
        })

        # Isi proxy OHLCV untuk hari ini dan tambahkan ke buffer
        proxy_open = history['Close'].iloc[-1]  # Open = Close kemarin
        proxy_high = pred_low * high_low_ratio
        proxy_close = pred_low * close_low_ratio

        new_row = pd.DataFrame([{
            'Time': forecast_date,
            'Open': proxy_open,
            'High': proxy_high,
            'Low': pred_low,
            'Close': proxy_close,
            'Volume': median_volume
        }])
        history = pd.concat([history, new_row], ignore_index=True)

    forecast_df = pd.DataFrame(predictions)

    # Print sampel
    print(f"\n  Hasil Forecast (Januari 2026):")
    print(f"  {'Tanggal':<12} {'Prediksi Low':>18} {'CI Lower':>18} {'CI Upper':>18}")
    print("  " + "-" * 68)
    for _, row in forecast_df.iterrows():
        print(f"  {row['Time'].strftime('%Y-%m-%d'):<12} "
              f"Rp {row['Predicted_Low']:>14,.0f} "
              f"Rp {row['Confidence_Lower']:>14,.0f} "
              f"Rp {row['Confidence_Upper']:>14,.0f}")

    print(f"\n  PERINGATAN: Recursive forecast degradasi seiring waktu.")
    print(f"  Hari 1-3 paling akurat (lag dari data asli).")
    print(f"  Setelah hari ~10, gunakan sebagai indikasi tren saja.")

    return forecast_df


# ============================================================================
# 8. EVALUASI FORECAST vs AKTUAL
# ============================================================================

def evaluate_forecast(forecast_df, eval_to_date):
    """
    Fetch data aktual dan bandingkan dengan forecast.

    Parameters:
    - forecast_df: hasil dari forecast_future()
    - eval_to_date: string "YYYY-MM-DD" untuk fetch data aktual

    Returns:
    - comparison DataFrame, metrics tuple
    """
    print(f"\n{'='*50}")
    print(f"  EVALUASI: Prediksi vs Aktual")
    print(f"{'='*50}")

    # Fetch data aktual
    actual_df = fetch_btc_data(to_date=eval_to_date)
    if actual_df is None:
        print("Gagal fetch data aktual untuk evaluasi")
        return None, None

    # Filter data aktual ke periode forecast
    forecast_start = forecast_df['Time'].min()
    forecast_end = forecast_df['Time'].max()

    actual_period = actual_df[
        (actual_df['Time'] >= forecast_start) &
        (actual_df['Time'] <= forecast_end)
    ].copy()

    if len(actual_period) == 0:
        print(f"Tidak ada data aktual untuk periode {forecast_start.date()} - {forecast_end.date()}")
        return None, None

    print(f"  Data aktual tersedia: {len(actual_period)} hari")

    # Join berdasarkan tanggal
    forecast_df['Date'] = forecast_df['Time'].dt.date
    actual_period['Date'] = actual_period['Time'].dt.date

    merged = pd.merge(
        forecast_df[['Date', 'Predicted_Low', 'Confidence_Lower', 'Confidence_Upper', 'Step']],
        actual_period[['Date', 'Low']].rename(columns={'Low': 'Actual_Low'}),
        on='Date',
        how='inner'
    )

    if len(merged) == 0:
        print("Tidak ada tanggal yang cocok antara forecast dan data aktual")
        return None, None

    # Hitung metrik
    actual = merged['Actual_Low'].values
    predicted = merged['Predicted_Low'].values

    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    r2 = r2_score(actual, predicted)

    # Cek berapa prediksi yang masuk confidence interval
    in_ci = ((merged['Actual_Low'] >= merged['Confidence_Lower']) &
             (merged['Actual_Low'] <= merged['Confidence_Upper'])).sum()
    ci_coverage = in_ci / len(merged) * 100

    print(f"\n  Metrik Evaluasi ({len(merged)} hari):")
    print(f"  MAPE : {mape:.2f}%", end="")
    if mape < 5:
        print("  [EXCELLENT]")
    elif mape < 10:
        print("  [GOOD]")
    elif mape < 15:
        print("  [ACCEPTABLE]")
    else:
        print("  [PERLU DITINGKATKAN]")
    print(f"  RMSE : Rp {rmse:,.0f}")
    print(f"  MAE  : Rp {mae:,.0f}")
    print(f"  R-sq : {r2:.4f}")
    print(f"  CI Coverage: {ci_coverage:.1f}% ({in_ci}/{len(merged)} hari dalam interval)")

    # Tabel perbandingan
    comparison = pd.DataFrame({
        'Tanggal': merged['Date'].apply(lambda x: x.strftime('%Y-%m-%d')),
        'Low Aktual': merged['Actual_Low'].apply(lambda x: f"Rp {x:,.0f}"),
        'Prediksi': merged['Predicted_Low'].apply(lambda x: f"Rp {x:,.0f}"),
        'Error (Rp)': (merged['Predicted_Low'] - merged['Actual_Low']).apply(lambda x: f"Rp {x:,.0f}"),
        'Error (%)': ((merged['Predicted_Low'] - merged['Actual_Low']).abs() / merged['Actual_Low'] * 100).apply(lambda x: f"{x:.2f}%"),
        'Dalam CI': ((merged['Actual_Low'] >= merged['Confidence_Lower']) &
                     (merged['Actual_Low'] <= merged['Confidence_Upper'])).apply(lambda x: "Ya" if x else "Tidak")
    })

    print(f"\n  Tabel Perbandingan:")
    print(comparison.to_string(index=False))

    metrics = {'mape': mape, 'rmse': rmse, 'mae': mae, 'r2': r2, 'ci_coverage': ci_coverage}
    return comparison, metrics, merged


# ============================================================================
# 9. VISUALISASI
# ============================================================================

def create_forecast_chart(df, forecast_df, save_path, merged_eval=None):
    """
    Chart forecast dengan confidence band dan optional evaluasi overlay
    """
    has_eval = merged_eval is not None and len(merged_eval) > 0

    n_plots = 3 if has_eval else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))

    # ===== Chart 1: Historis + Forecast + Confidence Band =====
    ax1 = axes[0]

    # 60 hari terakhir data historis
    context_start = df['Time'].max() - timedelta(days=60)
    recent = df[df['Time'] >= context_start]

    ax1.plot(recent['Time'], recent['Low'] / 1e9,
             label='Data Historis (Low)', color='#2196F3', linewidth=1.5)

    ax1.plot(forecast_df['Time'], forecast_df['Predicted_Low'] / 1e9,
             label='Forecast ARDL', color='#E91E63', linewidth=2, linestyle='--')

    ax1.fill_between(forecast_df['Time'],
                     forecast_df['Confidence_Lower'] / 1e9,
                     forecast_df['Confidence_Upper'] / 1e9,
                     alpha=0.2, color='#E91E63', label='95% Confidence Interval')

    # Garis pemisah
    ax1.axvline(x=df['Time'].max(), color='gray', linestyle='--',
                linewidth=1.5, label='Batas Data Historis')

    if has_eval:
        ax1.plot(pd.to_datetime(merged_eval['Date']),
                 merged_eval['Actual_Low'] / 1e9,
                 label='Aktual (Evaluasi)', color='#4CAF50', linewidth=1.5,
                 marker='o', markersize=3)

    ax1.set_title('Recursive Forecast Harga Low BTC/IDR - ARDL Model',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tanggal', fontsize=11)
    ax1.set_ylabel('Harga Low (Miliar IDR)', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # ===== Chart 2: Forecast saja (zoom in) =====
    ax2 = axes[1]

    ax2.plot(forecast_df['Time'], forecast_df['Predicted_Low'] / 1e9,
             label='Forecast', color='#E91E63', linewidth=2, marker='o', markersize=4)

    ax2.fill_between(forecast_df['Time'],
                     forecast_df['Confidence_Lower'] / 1e9,
                     forecast_df['Confidence_Upper'] / 1e9,
                     alpha=0.2, color='#E91E63', label='95% CI')

    if has_eval:
        ax2.plot(pd.to_datetime(merged_eval['Date']),
                 merged_eval['Actual_Low'] / 1e9,
                 label='Aktual', color='#4CAF50', linewidth=2,
                 marker='s', markersize=4)

    ax2.set_title('Detail Forecast (Zoom)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Tanggal', fontsize=11)
    ax2.set_ylabel('Harga Low (Miliar IDR)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # ===== Chart 3: Error distribution (jika ada evaluasi) =====
    if has_eval:
        ax3 = axes[2]
        error_pct = ((merged_eval['Predicted_Low'] - merged_eval['Actual_Low'])
                     / merged_eval['Actual_Low'] * 100)

        bars = ax3.bar(range(len(error_pct)), error_pct, color=['#4CAF50' if e >= 0 else '#F44336' for e in error_pct],
                       alpha=0.7, edgecolor='white')

        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.axhline(y=error_pct.mean(), color='#FF9800', linestyle='--',
                    linewidth=2, label=f'Mean: {error_pct.mean():.2f}%')

        ax3.set_title('Prediction Error per Hari (%)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Hari ke-', fontsize=11)
        ax3.set_ylabel('Error (%)', fontsize=11)
        ax3.set_xticks(range(0, len(error_pct), max(1, len(error_pct)//10)))
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart disimpan di: {save_path}")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  BTC/IDR LOW PRICE PREDICTION - PRODUCTION FORECASTING")
    print("  ARDL (Autoregressive Distributed Lags) Model v3")
    print("=" * 70)

    print(f"\n  Mode       : {MODE}")
    print(f"  Train data : sampai {TRAIN_DATA_END}")
    print(f"  Forecast   : {FORECAST_DAYS} hari ke depan")
    if MODE == "forecast_and_evaluate":
        print(f"  Eval data  : sampai {EVAL_DATA_END}")

    # 1. Fetch training data
    df = fetch_btc_data(to_date=TRAIN_DATA_END)
    if df is None:
        return

    # 2. ADF test
    adf_results = run_adf_test(df)

    # 3. Optimasi lag (AIC)
    p_lags, q_lags = select_optimal_lags(df, max_p=10, max_q=10, min_lag=MIN_LAG)

    # 4. Train model pada SELURUH data training
    ardl_model, df_features, features = train_ardl_model(
        df, p_lags, q_lags, min_lag=MIN_LAG
    )

    # 5. Recursive forecast
    forecast_df = forecast_future(
        ardl_model, df, features,
        n_days=FORECAST_DAYS,
        p_lags=p_lags, q_lags=q_lags, min_lag=MIN_LAG
    )

    # Simpan forecast ke CSV
    forecast_csv = f'{BASE_DIR}/prediction_ardl_v3_forecast_jan2026.csv'
    forecast_out = forecast_df[['Time', 'Predicted_Low', 'Confidence_Lower', 'Confidence_Upper']].copy()
    forecast_out['Time'] = forecast_out['Time'].dt.strftime('%Y-%m-%d')
    forecast_out['Predicted_Low'] = forecast_out['Predicted_Low'].apply(lambda x: f"Rp {x:,.0f}")
    forecast_out['Confidence_Lower'] = forecast_out['Confidence_Lower'].apply(lambda x: f"Rp {x:,.0f}")
    forecast_out['Confidence_Upper'] = forecast_out['Confidence_Upper'].apply(lambda x: f"Rp {x:,.0f}")
    forecast_out.to_csv(forecast_csv, index=False)
    print(f"\nForecast disimpan di: {forecast_csv}")

    # 6. Evaluasi (jika mode forecast_and_evaluate)
    merged_eval = None
    if MODE == "forecast_and_evaluate":
        comparison, metrics, merged_eval = evaluate_forecast(forecast_df, EVAL_DATA_END)

        if comparison is not None:
            eval_csv = f'{BASE_DIR}/prediction_ardl_v3_eval_jan2026.csv'
            comparison.to_csv(eval_csv, index=False)
            print(f"Evaluasi disimpan di: {eval_csv}")

    # 7. Visualisasi
    chart_path = f'{BASE_DIR}/prediction_ardl_v3_forecast_chart.png'
    create_forecast_chart(df, forecast_df, chart_path, merged_eval)

    # 8. Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Model: ARDL(p={p_lags}, q={q_lags}), min_lag={MIN_LAG}")
    print(f"  Regularization: Ridge (alpha={ardl_model.alpha})")
    print(f"  Features: {len(features)}")
    print(f"  Training data: sampai {TRAIN_DATA_END}")
    print(f"  Forecast: {FORECAST_DAYS} hari ({forecast_df['Time'].min().date()} - {forecast_df['Time'].max().date()})")

    if MODE == "forecast_and_evaluate" and merged_eval is not None:
        print(f"\n  Evaluasi vs Aktual:")
        print(f"    MAPE: {metrics['mape']:.2f}%")
        print(f"    RMSE: Rp {metrics['rmse']:,.0f}")
        print(f"    MAE:  Rp {metrics['mae']:,.0f}")
        print(f"    R-sq: {metrics['r2']:.4f}")
        print(f"    CI Coverage: {metrics['ci_coverage']:.1f}%")

    print(f"\n  Output files:")
    print(f"    Forecast CSV: {forecast_csv}")
    if MODE == "forecast_and_evaluate":
        print(f"    Eval CSV    : {BASE_DIR}/prediction_ardl_v3_eval_jan2026.csv")
    print(f"    Chart       : {chart_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
