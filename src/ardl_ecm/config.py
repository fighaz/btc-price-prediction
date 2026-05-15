"""
Konfigurasi default untuk engine ARDL-ECM monthly.

Nilai-nilai ini adalah default; fungsi engine menerima parameter agar
Streamlit / CLI bisa override per-run.
"""

# Lag selection (frekuensi bulanan -> lag kecil)
MAX_LAG_ENDOG = 3
MAX_LAG_EXOG = 3
IC = "aic"
TREND = "c"
VAR_MAXLAG = 3

# Transform & windowing
USE_LOG_TRANSFORM = True
ROLLING_WINDOW_YEARS = 0  # 0 = semua data sejak 2014

# Backtest
BACKTEST_MONTHS = 12

# Resample harian -> bulanan
MIN_DAYS_PER_MONTH = 20  # bulan dengan hari < ini dibuang saat drop_partial
