"""
Data acquisition & cleaning untuk BTC/IDR dari Indodax API
"""
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime

from config import INDODAX_API_URL, INDODAX_SYMBOL, INDODAX_TIMEFRAME, INDODAX_FROM_TS

logger = logging.getLogger(__name__)


def fetch_btc_data(to_date=None):
    """
    Fetch data BTC/IDR dari Indodax API.

    Parameters:
        to_date: string "YYYY-MM-DD" atau None (default 2025-12-31)

    Returns:
        DataFrame dengan kolom [Time, Open, High, Low, Close, Volume] atau None jika gagal
    """
    if to_date is None:
        to_ts = 1767200399  # 2025-12-31
        label = "2025-12-31"
    else:
        dt = datetime.strptime(to_date, "%Y-%m-%d")
        dt = dt.replace(hour=23, minute=59, second=59)
        to_ts = int(dt.timestamp())
        label = to_date

    params = {
        "from": INDODAX_FROM_TS,
        "to": to_ts,
        "symbol": INDODAX_SYMBOL,
        "tf": INDODAX_TIMEFRAME,
    }

    logger.info(f"Fetching data dari Indodax API (sampai {label})...")
    response = requests.get(INDODAX_API_URL, params=params)

    if response.status_code != 200:
        logger.error(f"Gagal fetch data: {response.status_code}")
        return None

    data = response.json()
    df = pd.DataFrame(data)

    df["Time"] = pd.to_datetime(df["Time"], unit="s")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)

    logger.info(f"Data mentah: {len(df)} records")
    logger.info(f"Rentang data: {df['Time'].min()} - {df['Time'].max()}")

    df = _clean_data(df)
    return df


def _clean_data(df):
    """Cleaning: dedup, sort, ffill, interpolasi, deteksi anomali"""
    before_dedup = len(df)
    df = df.drop_duplicates(subset="Time", keep="last").reset_index(drop=True)
    if before_dedup > len(df):
        logger.info(f"Duplikat dihapus: {before_dedup - len(df)} baris")

    df = df.sort_values("Time").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].ffill()
        df[col] = df[col].interpolate(method="linear")

    for col in ["Open", "High", "Low", "Close"]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3 * IQR
        upper = Q3 + 3 * IQR
        anomalies = ((df[col] < lower) | (df[col] > upper)).sum()
        if anomalies > 0:
            logger.info(f"Anomali di {col}: {anomalies} titik data (tidak dihapus)")

    logger.info(f"Data bersih: {len(df)} records")
    return df
