"""
Konfigurasi terpusat untuk BTC Price Prediction
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = DATA_DIR / "btc_prediction.db"

# API Indodax
INDODAX_API_URL = "https://indodax.com/tradingview/history_v2"
INDODAX_SYMBOL = "BTCIDR"
INDODAX_TIMEFRAME = "1D"
INDODAX_FROM_TS = 1391191321  # 2014-02-01

# ARDL Model defaults
DEFAULT_MIN_LAG = 2
DEFAULT_MAX_P = 10
DEFAULT_MAX_Q = 10
DEFAULT_FORECAST_DAYS = 31
ALPHA_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
CV_SPLITS = 5

# Database
DB_URL = f"sqlite:///{DB_PATH}"
