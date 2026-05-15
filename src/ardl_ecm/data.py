"""
Data acquisition & resampling untuk engine ARDL-ECM monthly.

Input data tetap harian dari Indodax, lalu di-resample ke bulanan:
    Open       = first day of month
    Close      = last day of month
    Low (Y)    = min of month  <- target prediksi
    High       = max of month
    Volume     = sum of month  -> log_Volume = log1p
"""
import logging
from datetime import datetime

import requests
import pandas as pd
import numpy as np

from config import (
    INDODAX_API_URL,
    INDODAX_SYMBOL,
    INDODAX_TIMEFRAME,
    INDODAX_FROM_TS,
)
from src.ardl_ecm.config import MIN_DAYS_PER_MONTH

logger = logging.getLogger(__name__)

_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def fetch_btc_daily(to_date=None):
    """
    Fetch data harian BTC/IDR dari Indodax API.

    Parameters:
        to_date: string "YYYY-MM-DD" atau None (default sampai hari ini)

    Returns:
        DataFrame harian dengan DatetimeIndex (freq D) dan kolom OHLCV.
    """
    if to_date is None:
        dt = datetime.now()
    else:
        dt = datetime.strptime(to_date, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59
        )

    to_ts = int(dt.timestamp())
    label = dt.strftime("%Y-%m-%d")

    params = {
        "from": INDODAX_FROM_TS,
        "to": to_ts,
        "symbol": INDODAX_SYMBOL,
        "tf": INDODAX_TIMEFRAME,
    }
    logger.info(f"Fetching BTC/IDR daily data (sampai {label})...")
    resp = requests.get(INDODAX_API_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal fetch data: HTTP {resp.status_code}")

    df = pd.DataFrame(resp.json())
    df["Time"] = pd.to_datetime(df["Time"], unit="s").dt.normalize()
    for col in _OHLCV:
        df[col] = df[col].astype(float)

    df = (
        df.drop_duplicates(subset="Time", keep="last")
        .sort_values("Time")
        .reset_index(drop=True)
    )
    for col in _OHLCV:
        df[col] = df[col].replace(0, np.nan).ffill().interpolate(method="linear")

    df = df.set_index("Time").asfreq("D")
    for col in _OHLCV:
        df[col] = df[col].ffill()

    logger.info(
        f"Data harian bersih: {len(df)} records "
        f"({df.index.min().date()} - {df.index.max().date()})"
    )
    return df


def resample_to_monthly(df_daily, drop_partial=True, min_days=MIN_DAYS_PER_MONTH):
    """
    Agregasi data harian ke bulanan.

    drop_partial=True: bulan terakhir dibuang kalau jumlah hari < min_days.

    Returns:
        DataFrame bulanan dengan kolom Open/High/Low/Close/Volume/log_Volume.
    """
    monthly = pd.DataFrame(
        {
            "Open": df_daily["Open"].resample("M").first(),
            "High": df_daily["High"].resample("M").max(),
            "Low": df_daily["Low"].resample("M").min(),
            "Close": df_daily["Close"].resample("M").last(),
            "Volume": df_daily["Volume"].resample("M").sum(),
        }
    )

    if drop_partial:
        days_per_month = df_daily["Low"].resample("M").count()
        monthly = monthly[days_per_month >= min_days]

    monthly["log_Volume"] = np.log1p(monthly["Volume"])
    monthly = monthly.dropna()
    logger.info(
        f"Data bulanan: {len(monthly)} bulan "
        f"({monthly.index.min().strftime('%Y-%m')} - "
        f"{monthly.index.max().strftime('%Y-%m')})"
    )
    return monthly
