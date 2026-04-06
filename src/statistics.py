"""
Uji statistik untuk validasi data time series
"""
import logging
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def run_adf_test(df):
    """
    Augmented Dickey-Fuller (ADF) Test untuk stasioneritas.
    ARDL valid untuk campuran I(0) dan I(1).

    Returns:
        dict dengan hasil ADF per kolom, termasuk first difference jika non-stasioner
    """
    columns_to_test = ["Low", "Open", "High", "Close", "Volume"]
    results = {}

    for col in columns_to_test:
        result = adfuller(df[col].dropna(), autolag="AIC")
        adf_stat, p_value = result[0], result[1]
        is_stationary = p_value < 0.05

        results[col] = {
            "adf_statistic": adf_stat,
            "p_value": p_value,
            "stationary": is_stationary,
            "status": "Stasioner I(0)" if is_stationary else "Non-stasioner",
        }

    # First difference test untuk variabel non-stasioner
    non_stationary = [col for col, r in results.items() if not r["stationary"]]
    for col in non_stationary:
        diff_result = adfuller(df[col].diff().dropna(), autolag="AIC")
        diff_p = diff_result[1]
        results[col]["diff_adf_statistic"] = diff_result[0]
        results[col]["diff_p_value"] = diff_p
        results[col]["diff_stationary"] = diff_p < 0.05
        results[col]["diff_status"] = "I(1) - OK" if diff_p < 0.05 else "I(2) - WARNING"

    logger.info(f"ADF Test selesai: {len(non_stationary)} variabel non-stasioner")
    return results
