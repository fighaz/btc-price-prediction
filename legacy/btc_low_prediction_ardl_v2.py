"""
BTC/IDR Low Price Prediction
Menggunakan Metode ARDL (Autoregressive Distributed Lags)

Memprediksi harga LOW Bitcoin menggunakan data historis dari Indodax.
Data dibagi 80:20 secara sekuensial (kronologis).

ARDL Model: Y_t = c + Sigma(phi_i * Y_{t-i}) + Sigma(beta_j * X_{t-j}) + epsilon_t
- Autoregressive: menggunakan lag dari variabel dependen (Low price)
- Distributed Lags: menggunakan lag dari variabel independen (Open, High, Close, Volume)

Metodologi:
1. Akuisisi data dari Indodax API (2014 - 2025-12-31)
2. Preprocessing: dedup, forward fill, interpolasi, validasi anomali
3. Feature Engineering: AR lags, DL lags, momentum, volatilitas
4. Uji Stasioneritas: ADF (Augmented Dickey-Fuller) Test
5. Optimasi Parameter: AIC (Akaike Information Criterion)
6. Training: OLS/Ridge dengan Time Series Cross-Validation
7. Evaluasi: MAPE, RMSE, MAE, R-squared
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA ACQUISITION
# ============================================================================

def fetch_btc_data():
    """
    Fetch data BTC/IDR dari Indodax API
    Data dari awal (2014) sampai 2025-12-31
    """
    url = "https://indodax.com/tradingview/history_v2"

    params = {
        "from": 1391191321,    # timestamp awal (2014-02-01)
        "to": 1767200399,      # timestamp akhir (2025-12-31)
        "symbol": "BTCIDR",    # pair yang diambil
        "tf": "1D"             # timeframe 1 hari
    }

    print("Fetching data dari Indodax API...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Gagal fetch data: {response.status_code}")
        return None

    data = response.json()
    df = pd.DataFrame(data)

    # Convert UNIX timestamp -> datetime
    df["Time"] = pd.to_datetime(df["Time"], unit="s")

    # Convert to float
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    print(f"Data mentah: {len(df)} records")
    print(f"Rentang data: {df['Time'].min()} - {df['Time'].max()}")

    # --- Data Cleaning (Preprocessing) ---

    # 1. Hapus duplikat berdasarkan timestamp
    before_dedup = len(df)
    df = df.drop_duplicates(subset='Time', keep='last').reset_index(drop=True)
    if before_dedup > len(df):
        print(f"Duplikat dihapus: {before_dedup - len(df)} baris")

    # 2. Sort kronologis
    df = df.sort_values('Time').reset_index(drop=True)

    # 3. Handle missing values - forward fill lalu interpolasi linear
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].replace(0, np.nan)  # Treat 0 sebagai missing untuk harga
        df[col] = df[col].ffill()              # Forward fill
        df[col] = df[col].interpolate(method='linear')  # Interpolasi sisa

    # 4. Validasi anomali (lapor, jangan hapus - harga BTC memang volatile)
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
    """
    Augmented Dickey-Fuller (ADF) Test untuk stasioneritas.
    Memastikan tidak ada variabel yang terintegrasi pada ordo kedua I(2).
    ARDL valid untuk campuran I(0) dan I(1).
    """
    print("\nAugmented Dickey-Fuller (ADF) Stationarity Test:")
    print("-" * 65)
    print(f"  {'Variabel':<12} {'ADF Stat':>10} {'p-value':>10} {'Status':<15}")
    print("-" * 65)

    columns_to_test = ['Low', 'Open', 'High', 'Close', 'Volume']
    results = {}

    for col in columns_to_test:
        result = adfuller(df[col].dropna(), autolag='AIC')
        adf_stat = result[0]
        p_value = result[1]
        is_stationary = p_value < 0.05
        status = "Stasioner I(0)" if is_stationary else "Non-stasioner"

        results[col] = {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'stationary': is_stationary
        }

        print(f"  {col:<12} {adf_stat:>10.4f} {p_value:>10.4f} {status:<15}")

    # Test first differences untuk variabel non-stasioner
    non_stationary = [col for col, r in results.items() if not r['stationary']]
    if non_stationary:
        print("\n  First Difference Test (variabel non-stasioner):")
        for col in non_stationary:
            diff_result = adfuller(df[col].diff().dropna(), autolag='AIC')
            diff_p = diff_result[1]
            diff_status = "I(1) - OK" if diff_p < 0.05 else "I(2) - WARNING"
            print(f"  d({col}): ADF={diff_result[0]:.4f}, p={diff_p:.4f} -> {diff_status}")
            if diff_p >= 0.05:
                print(f"    PERINGATAN: {col} mungkin I(2), ARDL kurang sesuai")

    print("-" * 65)
    print("  Catatan: ARDL valid untuk campuran variabel I(0) dan I(1)")

    return results


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def prepare_ardl_features(df, p_lags=7, q_lags=7, min_lag=2):
    """
    Feature engineering untuk ARDL model (anti-overfitting version)

    Parameters:
    - p_lags: jumlah lag untuk variabel dependen (Low) - Autoregressive
    - q_lags: jumlah lag untuk variabel independen (Open, High, Close) - Distributed Lags
    - min_lag: minimum lag offset untuk mencegah data leakage (default=2)

    ARDL(p, q1, q2, ...) model formula:
    Low_t = c + Sigma(phi_i * Low_{t-i}) + Sigma(beta1_j * Open_{t-j}) + ...

    Anti-overfitting measures:
    - Minimum lag = 2 (bukan 1) agar model tidak hanya menyalin harga kemarin
    - Moving averages & EMA dihitung dari variabel INDEPENDEN (Open, High, Close),
      BUKAN dari variabel target (Low), untuk mencegah data leakage
    - Volatilitas dihitung dari spread (High-Low), bukan dari Low langsung
    - Gunakan percentage return, bukan absolute price difference
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    df['DayIndex'] = range(len(df))

    # ===== AUTOREGRESSIVE COMPONENT (Lag variabel dependen: Low) =====
    # Dimulai dari min_lag untuk mencegah "copy harga kemarin"
    for i in range(min_lag, min_lag + p_lags):
        df[f'Low_lag{i}'] = df['Low'].shift(i)

    # ===== DISTRIBUTED LAGS COMPONENT (Lag variabel independen) =====
    # Juga dimulai dari min_lag untuk konsistensi

    for i in range(min_lag, min_lag + q_lags):
        df[f'Open_lag{i}'] = df['Open'].shift(i)

    for i in range(min_lag, min_lag + q_lags):
        df[f'High_lag{i}'] = df['High'].shift(i)

    for i in range(min_lag, min_lag + q_lags):
        df[f'Close_lag{i}'] = df['Close'].shift(i)

    # Lag for Volume (log-transformed for stability)
    df['Log_Volume'] = np.log1p(df['Volume'])
    for i in range(min_lag, min_lag + q_lags):
        df[f'LogVolume_lag{i}'] = df['Log_Volume'].shift(i)

    # ===== MOMENTUM FEATURES =====
    # Percentage returns (bukan absolute diff - lebih bermakna & scale-invariant)
    df['Low_pct'] = df['Low'].pct_change().shift(min_lag)
    df['Open_pct'] = df['Open'].pct_change().shift(min_lag)
    df['High_pct'] = df['High'].pct_change().shift(min_lag)
    df['Close_pct'] = df['Close'].pct_change().shift(min_lag)

    # Moving averages dari variabel INDEPENDEN (bukan dari Low/target)
    # Ini menangkap tren pasar tanpa leakage ke target
    df['MA7_Close'] = df['Close'].rolling(window=7).mean().shift(min_lag)
    df['MA14_Close'] = df['Close'].rolling(window=14).mean().shift(min_lag)
    df['MA30_Close'] = df['Close'].rolling(window=30).mean().shift(min_lag)

    # EMA dari Close (bukan Low)
    df['EMA7_Close'] = df['Close'].ewm(span=7, adjust=False).mean().shift(min_lag)
    df['EMA14_Close'] = df['Close'].ewm(span=14, adjust=False).mean().shift(min_lag)

    # ===== VOLATILITY FEATURES =====
    # Volatilitas dari spread (High-Low), bukan dari Low langsung
    df['HL_Spread'] = df['High'] - df['Low']
    df['Volatility_7d'] = df['HL_Spread'].rolling(window=7).std().shift(min_lag)
    df['Volatility_14d'] = df['HL_Spread'].rolling(window=14).std().shift(min_lag)

    # Lagged spreads
    for i in range(min_lag, min_lag + 3):
        df[f'HL_Spread_lag{i}'] = df['HL_Spread'].shift(i)

    df['OC_Spread'] = df['Close'] - df['Open']
    for i in range(min_lag, min_lag + 3):
        df[f'OC_Spread_lag{i}'] = df['OC_Spread'].shift(i)

    # Drop rows with NaN (due to lagging)
    df = df.dropna().reset_index(drop=True)

    return df


def get_ardl_features(p_lags=7, q_lags=7, min_lag=2):
    """
    Get list of feature names untuk ARDL model
    """
    features = []

    # Autoregressive lags (Low) - mulai dari min_lag
    for i in range(min_lag, min_lag + p_lags):
        features.append(f'Low_lag{i}')

    # Distributed lags (Open, High, Close, Volume) - mulai dari min_lag
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'Open_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'High_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'Close_lag{i}')
    for i in range(min_lag, min_lag + q_lags):
        features.append(f'LogVolume_lag{i}')

    # Momentum features (percentage returns, shifted)
    features.extend([
        'Low_pct', 'Open_pct', 'High_pct', 'Close_pct',
    ])

    # Moving averages & EMA dari Close (bukan Low)
    features.extend([
        'MA7_Close', 'MA14_Close', 'MA30_Close',
        'EMA7_Close', 'EMA14_Close',
    ])

    # Volatility
    features.extend(['Volatility_7d', 'Volatility_14d'])

    # Spread features
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
    Select optimal lag order menggunakan AIC (Akaike Information Criterion)

    AIC = n * ln(RSS/n) + 2k
    where:
    - n = number of observations
    - RSS = residual sum of squares
    - k = number of parameters

    min_lag=2 untuk mencegah overfitting (tidak menggunakan lag-1)
    """
    print("\nMencari optimal lag order menggunakan AIC...")
    print(f"  (minimum lag offset = {min_lag} untuk mencegah overfitting)")

    best_aic = np.inf
    best_p = 1
    best_q = 1

    results = []

    for p in range(1, max_p + 1, 2):  # Step by 2 untuk efisiensi
        for q in range(1, max_q + 1, 2):
            try:
                temp_df = prepare_ardl_features(df.copy(), p_lags=p, q_lags=q, min_lag=min_lag)

                if len(temp_df) < 100:
                    continue

                # Gunakan 80% data untuk seleksi lag
                train_size = int(len(temp_df) * 0.8)
                train_df = temp_df.iloc[:train_size]

                features = get_ardl_features(p, q, min_lag=min_lag)
                available_features = [f for f in features if f in train_df.columns]

                X = train_df[available_features].values
                y = train_df['Low'].values

                # Fit OLS
                model = LinearRegression()
                model.fit(X, y)

                # Calculate AIC
                y_pred = model.predict(X)
                n = len(y)
                k = X.shape[1] + 1  # features + intercept
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

    # Tampilkan top 5 kombinasi
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
    """
    Autoregressive Distributed Lag (ARDL) Model

    Menggabungkan:
    1. Autoregressive (AR): lag dari variabel target (Low)
    2. Distributed Lags (DL): lag dari variabel prediktor (Open, High, Close, Volume)

    Kelebihan ARDL:
    - Tidak memerlukan stasioneritas data (berbeda dengan VAR/VARMA)
    - Dapat menangani variabel dengan ordo integrasi berbeda I(0) atau I(1)
    - Menghasilkan estimasi long-run dan short-run yang konsisten
    """

    def __init__(self, p_lags=7, q_lags=7, regularization='ridge', alpha=1.0):
        self.p_lags = p_lags
        self.q_lags = q_lags
        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names

        # Standardize features (parameter mean/std hanya dari training data)
        X_scaled = self.scaler.fit_transform(X)

        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.alpha)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.alpha)
        else:
            self.model = LinearRegression()

        self.model.fit(X_scaled, y)

        if hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_)

        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_top_features(self, n=10):
        if self.feature_importance is None or self.feature_names is None:
            return []
        indices = np.argsort(self.feature_importance)[::-1][:n]
        return [(self.feature_names[i], self.feature_importance[i]) for i in indices]

    def get_model_summary(self):
        return {
            'p_lags': self.p_lags,
            'q_lags': self.q_lags,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'intercept': self.model.intercept_ if hasattr(self.model, 'intercept_') else None
        }


# ============================================================================
# 6. TRAINING & PREDICTION
# ============================================================================

def train_ardl_model(df, p_lags=7, q_lags=7, min_lag=2):
    """
    Training ARDL model dengan 80:20 sequential split
    """
    # Prepare features
    df_features = prepare_ardl_features(df.copy(), p_lags, q_lags, min_lag=min_lag)

    # 80:20 sequential split (TIDAK boleh random shuffle pada time series)
    split_idx = int(len(df_features) * 0.8)

    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()

    print(f"\nData Split: 80:20 Sequential")
    print(f"  Training: {len(train_df)} samples ({train_df['Time'].min().date()} - {train_df['Time'].max().date()})")
    print(f"  Testing:  {len(test_df)} samples ({test_df['Time'].min().date()} - {test_df['Time'].max().date()})")

    # Get features
    features = get_ardl_features(p_lags, q_lags, min_lag=min_lag)
    available_features = [f for f in features if f in train_df.columns]

    print(f"\nTraining ARDL Model...")
    print(f"  p-lags (AR): {p_lags}")
    print(f"  q-lags (DL): {q_lags}")
    print(f"  min_lag offset: {min_lag}")
    print(f"  Total features: {len(available_features)}")

    X_train = train_df[available_features].values
    y_train = train_df['Low'].values

    # Cross-validation untuk memilih alpha terbaik (expanding window)
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha = 1.0
    best_cv_score = -np.inf

    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
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

    # Train final ARDL model
    ardl_model = ARDLModel(p_lags=p_lags, q_lags=q_lags, regularization='ridge', alpha=best_alpha)
    ardl_model.fit(X_train, y_train, available_features)

    # Evaluate on training data
    y_train_pred = ardl_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"  R-squared (training): {train_r2:.4f}")

    # Show top features
    print("\n  Top 10 Most Important Features:")
    for feat, importance in ardl_model.get_top_features(10):
        print(f"    - {feat}: {importance:.4f}")

    return ardl_model, df_features, split_idx, available_features


def predict_with_ardl(df_features, split_idx, ardl_model, features):
    """
    Membuat prediksi untuk test set (20% data terakhir) menggunakan ARDL
    """
    test_df = df_features.iloc[split_idx:].copy()

    if len(test_df) == 0:
        print("Test data tidak tersedia")
        return None

    test_period = f"{test_df['Time'].min().date()} - {test_df['Time'].max().date()}"
    print(f"\nPrediksi ARDL untuk test period: {test_period} ({len(test_df)} hari)")

    X_test = test_df[features].values
    predictions = ardl_model.predict(X_test)

    test_df['Pred_ARDL'] = predictions

    return test_df


# ============================================================================
# 7. EVALUATION
# ============================================================================

def calculate_metrics(actual, predicted, model_name):
    """
    Menghitung metrik evaluasi: MAPE, RMSE, MAE, R-squared
    """
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = np.mean(np.abs(actual - predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{model_name}:")
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

    return mape, rmse, mae, r2


# ============================================================================
# 8. OUTPUT (CSV & CHART)
# ============================================================================

def create_comparison_table(test_df):
    """
    Membuat tabel perbandingan harga aktual vs prediksi
    """
    comparison = pd.DataFrame({
        'Tanggal': test_df['Time'].dt.strftime('%Y-%m-%d'),
        'Low Aktual': test_df['Low'].apply(lambda x: f"Rp {x:,.0f}"),
        'Prediksi ARDL': test_df['Pred_ARDL'].apply(lambda x: f"Rp {x:,.0f}"),
        'Error (Rp)': (test_df['Pred_ARDL'] - test_df['Low']).apply(lambda x: f"Rp {x:,.0f}"),
        'Error (%)': ((test_df['Pred_ARDL'] - test_df['Low']).abs() / test_df['Low'] * 100).apply(lambda x: f"{x:.2f}%")
    })

    return comparison


def create_visualizations(test_df, full_df, save_path):
    """
    Membuat chart perbandingan actual vs predicted
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # ===== Chart 1: Actual vs Predicted (Test Period) =====
    ax1 = axes[0]

    ax1.plot(test_df['Time'], test_df['Low'] / 1e9,
             label='Actual Low', color='#2196F3', linewidth=1.5)
    ax1.plot(test_df['Time'], test_df['Pred_ARDL'] / 1e9,
             label='ARDL Prediction', color='#E91E63', linewidth=1.5)

    ax1.fill_between(test_df['Time'],
                     test_df['Low'] / 1e9,
                     test_df['Pred_ARDL'] / 1e9,
                     alpha=0.2, color='gray', label='Prediction Gap')

    test_start = test_df['Time'].min().date()
    test_end = test_df['Time'].max().date()
    ax1.set_title(f'Prediksi Harga Low BTC/IDR - ARDL Model\nTest Period: {test_start} s/d {test_end}',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Tanggal', fontsize=11)
    ax1.set_ylabel('Harga Low (Miliar IDR)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # ===== Chart 2: Prediction Error Distribution =====
    ax2 = axes[1]

    error_pct = ((test_df['Pred_ARDL'] - test_df['Low']) / test_df['Low'] * 100)

    # Gunakan histogram karena test set besar
    ax2.hist(error_pct, bins=50, color='#2196F3', alpha=0.7, edgecolor='white')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=error_pct.mean(), color='#FF9800', linestyle='--',
                linewidth=2, label=f'Mean Error: {error_pct.mean():.2f}%')
    ax2.axvline(x=error_pct.median(), color='#4CAF50', linestyle='--',
                linewidth=2, label=f'Median Error: {error_pct.median():.2f}%')

    ax2.set_title('Distribusi Prediction Error (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Error (%)', fontsize=11)
    ax2.set_ylabel('Frekuensi', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # ===== Chart 3: Historical context with prediction =====
    ax3 = axes[2]

    # Tampilkan 60 hari sebelum split + seluruh test set
    context_start = test_df['Time'].min() - timedelta(days=60)
    recent_df = full_df[full_df['Time'] >= context_start]

    ax3.plot(recent_df['Time'], recent_df['Low'] / 1e9,
             label='Historical Low', color='#2196F3', linewidth=1.5, alpha=0.7)

    ax3.plot(test_df['Time'], test_df['Pred_ARDL'] / 1e9,
             label='ARDL Prediction (Test Period)', color='#E91E63', linewidth=1.5)

    # Garis pemisah training/test
    ax3.axvline(x=test_df['Time'].iloc[0], color='gray', linestyle='--',
                linewidth=1.5, label='Training/Test Split')

    ax3.set_title('Konteks Historis dengan Prediksi ARDL', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Tanggal', fontsize=11)
    ax3.set_ylabel('Harga Low (Miliar IDR)', fontsize=11)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart disimpan di: {save_path}")

    plt.show()


# ============================================================================
# PENJELASAN METODE ARDL
# ============================================================================

def explain_ardl():
    """
    Penjelasan tentang metode ARDL
    """
    explanation = """
====================================================================
         AUTOREGRESSIVE DISTRIBUTED LAGS (ARDL)
====================================================================

  Model ARDL menggabungkan dua konsep utama:

  1. AUTOREGRESSIVE (AR):
     - Menggunakan nilai masa lalu dari variabel target (Low price)
     - Contoh: Low_t = f(Low_{t-1}, Low_{t-2}, ..., Low_{t-p})

  2. DISTRIBUTED LAGS (DL):
     - Menggunakan nilai masa lalu dari variabel independen
     - Contoh: Open_{t-1}, High_{t-1}, Close_{t-1}, dst.

  Formula ARDL(p, q1, q2, ...):
  Y_t = c + Sigma(phi_i * Y_{t-i}) + Sigma(beta_j * X_{t-j}) + eps_t

  Kelebihan ARDL:
  - Tidak memerlukan data stasioner (berbeda dengan VAR/ARIMA)
  - Dapat mixing variabel I(0) dan I(1)
  - Estimasi short-run dan long-run yang konsisten
  - Cocok untuk data time series dengan multiple predictors

  Implementasi ini menggunakan:
  - p lags untuk Low price (autoregressive component)
  - q lags untuk Open, High, Close, Volume (distributed lags)
  - Ridge regularization untuk mencegah overfitting
  - AIC untuk pemilihan optimal lag order
  - StandardScaler normalisasi (fit hanya pada training data)
  - 80:20 sequential split (tanpa random shuffling)

  Anti-Overfitting Measures:
  - Minimum lag = 2 (bukan 1), mencegah model hanya menyalin harga kemarin
  - Moving averages dihitung dari Close (independen), bukan dari Low (target)
  - Volatilitas dihitung dari spread, bukan dari target langsung
  - Percentage returns, bukan absolute price differences

====================================================================
"""
    print(explanation)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main function - menjalankan pipeline lengkap
    """
    print("=" * 70)
    print("  BTC/IDR LOW PRICE PREDICTION")
    print("  Using ARDL (Autoregressive Distributed Lags) Model")
    print("=" * 70)

    # Penjelasan metode
    explain_ardl()

    # 1. Fetch & preprocess data
    df = fetch_btc_data()
    if df is None:
        return

    # 2. Uji stasioneritas (ADF Test)
    adf_results = run_adf_test(df)

    # 3. Optimasi lag order (AIC)
    # min_lag=2: mencegah overfitting dengan tidak menggunakan lag-1
    # yang terlalu dekat dengan target dan menyebabkan model hanya "menyalin"
    min_lag = 2
    p_lags, q_lags = select_optimal_lags(df, max_p=10, max_q=10, min_lag=min_lag)

    # 4. Train ARDL model (80:20 split)
    ardl_model, df_features, split_idx, features = train_ardl_model(
        df, p_lags, q_lags, min_lag=min_lag
    )

    # 5. Prediksi pada test set
    test_df = predict_with_ardl(df_features, split_idx, ardl_model, features)
    if test_df is None:
        return

    # 6. Evaluasi performa
    actual = test_df['Low'].values

    print("\n" + "=" * 50)
    print("  EVALUATION METRICS")
    print("=" * 50)

    mape, rmse, mae, r2 = calculate_metrics(
        actual, test_df['Pred_ARDL'].values, "ARDL Model"
    )

    # 7. Tabel perbandingan
    print("\n" + "=" * 50)
    print("  TABEL PERBANDINGAN (sampel 10 baris pertama & terakhir)")
    print("=" * 50)

    comparison = create_comparison_table(test_df)

    # Tampilkan sampel (test set bisa ratusan hari)
    if len(comparison) > 20:
        print("\n--- 10 baris pertama ---")
        print(comparison.head(10).to_string(index=False))
        print(f"\n... ({len(comparison) - 20} baris lainnya) ...")
        print("\n--- 10 baris terakhir ---")
        print(comparison.tail(10).to_string(index=False))
    else:
        print("\n" + comparison.to_string(index=False))

    # Save to CSV
    csv_path = '/home/fighaz/Documents/btc-price-prediction/prediction_ardl_v2_comparison.csv'
    comparison.to_csv(csv_path, index=False)
    print(f"\nTabel lengkap disimpan di: {csv_path}")

    # 8. Visualisasi
    chart_path = '/home/fighaz/Documents/btc-price-prediction/prediction_ardl_v2_chart.png'
    create_visualizations(test_df, df_features, chart_path)

    # 9. Summary
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)

    print(f"\n  ARDL Model Performance:")
    print(f"    MAPE : {mape:.2f}%")
    print(f"    RMSE : Rp {rmse:,.0f}")
    print(f"    MAE  : Rp {mae:,.0f}")
    print(f"    R-sq : {r2:.4f}")

    print(f"\n  Model Configuration:")
    print(f"    AR Lags (p)    : {p_lags}")
    print(f"    DL Lags (q)    : {q_lags}")
    print(f"    Total Features : {len(features)}")
    print(f"    Regularization : Ridge (alpha={ardl_model.alpha})")
    print(f"    Data Split     : 80:20 Sequential")

    if mape < 5:
        print("\n  Performa EXCELLENT (MAPE < 5%)")
    elif mape < 10:
        print("\n  Performa GOOD (MAPE < 10%)")
    elif mape < 15:
        print("\n  Performa ACCEPTABLE (MAPE < 15%)")
    else:
        print("\n  Performa perlu ditingkatkan (MAPE >= 15%)")

    print("\n" + "=" * 70)
    print("  ARDL Model selesai! Output files:")
    print(f"    Chart: {chart_path}")
    print(f"    CSV  : {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
