"""
Konfigurasi terpusat untuk BTC Price Prediction.

Konfigurasi database dibaca dari environment variable (lihat .env.example).
Salin .env.example -> .env lalu sesuaikan. Mendukung backend SQLite & MySQL.
"""
import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Paths
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = DATA_DIR / "btc_prediction.db"

# ============================================================================
# API Indodax
# ============================================================================
INDODAX_API_URL = "https://indodax.com/tradingview/history_v2"
INDODAX_SYMBOL = "BTCIDR"
INDODAX_TIMEFRAME = "1D"
INDODAX_FROM_TS = 1391191321  # 2014-02-01

# ============================================================================
# Database — backend konfigurabel via env
# ============================================================================
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").strip().lower()


def _build_db_url():
    """Bangun SQLAlchemy URL sesuai DB_BACKEND."""
    if DB_BACKEND == "mysql":
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "root")
        password = quote_plus(os.getenv("MYSQL_PASSWORD", ""))
        database = os.getenv("MYSQL_DATABASE", "btc_prediction")
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    # default: sqlite
    return f"sqlite:///{DB_PATH}"


DB_URL = _build_db_url()
