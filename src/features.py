"""
Feature engineering untuk ARDL model
"""
import numpy as np
import pandas as pd


def prepare_ardl_features(df, p_lags=7, q_lags=7, min_lag=2):
    """
    Feature engineering untuk ARDL model (anti-overfitting).

    Anti-overfitting measures:
    - Minimum lag = 2 (bukan 1) agar model tidak menyalin harga kemarin
    - MA & EMA dari Close (independen), bukan Low (target)
    - Volatilitas dari spread (High-Low), bukan Low
    - Percentage return, bukan absolute diff
    """
    df = df.copy()
    df = df.reset_index(drop=True)
    df["DayIndex"] = range(len(df))

    # AR lags (Low)
    for i in range(min_lag, min_lag + p_lags):
        df[f"Low_lag{i}"] = df["Low"].shift(i)

    # DL lags (Open, High, Close, Volume)
    for i in range(min_lag, min_lag + q_lags):
        df[f"Open_lag{i}"] = df["Open"].shift(i)
    for i in range(min_lag, min_lag + q_lags):
        df[f"High_lag{i}"] = df["High"].shift(i)
    for i in range(min_lag, min_lag + q_lags):
        df[f"Close_lag{i}"] = df["Close"].shift(i)

    df["Log_Volume"] = np.log1p(df["Volume"])
    for i in range(min_lag, min_lag + q_lags):
        df[f"LogVolume_lag{i}"] = df["Log_Volume"].shift(i)

    # Momentum (percentage returns)
    df["Low_pct"] = df["Low"].pct_change().shift(min_lag)
    df["Open_pct"] = df["Open"].pct_change().shift(min_lag)
    df["High_pct"] = df["High"].pct_change().shift(min_lag)
    df["Close_pct"] = df["Close"].pct_change().shift(min_lag)

    # MA & EMA dari Close
    df["MA7_Close"] = df["Close"].rolling(window=7).mean().shift(min_lag)
    df["MA14_Close"] = df["Close"].rolling(window=14).mean().shift(min_lag)
    df["MA30_Close"] = df["Close"].rolling(window=30).mean().shift(min_lag)
    df["EMA7_Close"] = df["Close"].ewm(span=7, adjust=False).mean().shift(min_lag)
    df["EMA14_Close"] = df["Close"].ewm(span=14, adjust=False).mean().shift(min_lag)

    # Volatilitas dari spread
    df["HL_Spread"] = df["High"] - df["Low"]
    df["Volatility_7d"] = df["HL_Spread"].rolling(window=7).std().shift(min_lag)
    df["Volatility_14d"] = df["HL_Spread"].rolling(window=14).std().shift(min_lag)

    for i in range(min_lag, min_lag + 3):
        df[f"HL_Spread_lag{i}"] = df["HL_Spread"].shift(i)

    df["OC_Spread"] = df["Close"] - df["Open"]
    for i in range(min_lag, min_lag + 3):
        df[f"OC_Spread_lag{i}"] = df["OC_Spread"].shift(i)

    df = df.dropna().reset_index(drop=True)
    return df


def get_ardl_features(p_lags=7, q_lags=7, min_lag=2):
    """Get list of feature names untuk ARDL model"""
    features = []

    for i in range(min_lag, min_lag + p_lags):
        features.append(f"Low_lag{i}")
    for i in range(min_lag, min_lag + q_lags):
        features.append(f"Open_lag{i}")
    for i in range(min_lag, min_lag + q_lags):
        features.append(f"High_lag{i}")
    for i in range(min_lag, min_lag + q_lags):
        features.append(f"Close_lag{i}")
    for i in range(min_lag, min_lag + q_lags):
        features.append(f"LogVolume_lag{i}")

    features.extend(["Low_pct", "Open_pct", "High_pct", "Close_pct"])
    features.extend(["MA7_Close", "MA14_Close", "MA30_Close", "EMA7_Close", "EMA14_Close"])
    features.extend(["Volatility_7d", "Volatility_14d"])

    for i in range(min_lag, min_lag + 3):
        features.append(f"HL_Spread_lag{i}")
    for i in range(min_lag, min_lag + 3):
        features.append(f"OC_Spread_lag{i}")

    return features


def compute_single_row_features(history_df, p_lags, q_lags, min_lag, feature_names):
    """
    Hitung feature vector untuk SATU baris terakhir dari history buffer.
    Digunakan untuk recursive forecasting.

    Parameters:
        history_df: DataFrame dengan kolom [Time, Open, High, Low, Close, Volume]
        p_lags, q_lags, min_lag: parameter ARDL
        feature_names: list nama fitur yang dibutuhkan model

    Returns:
        numpy array (1, n_features)
    """
    df = history_df.copy()
    df = df.reset_index(drop=True)
    last_idx = len(df) - 1

    feature_values = {}

    # AR lags (Low)
    for i in range(min_lag, min_lag + p_lags):
        col = f"Low_lag{i}"
        if col in feature_names:
            feature_values[col] = df["Low"].iloc[last_idx - i]

    # DL lags
    for var_name, col_name in [("Open", "Open"), ("High", "High"), ("Close", "Close")]:
        for i in range(min_lag, min_lag + q_lags):
            fname = f"{var_name}_lag{i}"
            if fname in feature_names:
                feature_values[fname] = df[col_name].iloc[last_idx - i]

    # LogVolume lags
    log_vol = np.log1p(df["Volume"].values)
    for i in range(min_lag, min_lag + q_lags):
        fname = f"LogVolume_lag{i}"
        if fname in feature_names:
            feature_values[fname] = log_vol[last_idx - i]

    # Percentage returns (shifted by min_lag)
    for var_name in ["Low", "Open", "High", "Close"]:
        fname = f"{var_name}_pct"
        if fname in feature_names:
            idx = last_idx - min_lag
            if idx > 0:
                prev = df[var_name].iloc[idx - 1]
                curr = df[var_name].iloc[idx]
                feature_values[fname] = (curr - prev) / prev if prev != 0 else 0
            else:
                feature_values[fname] = 0

    # Moving averages dari Close (shifted by min_lag)
    close_vals = df["Close"].values
    for window, fname in [(7, "MA7_Close"), (14, "MA14_Close"), (30, "MA30_Close")]:
        if fname in feature_names:
            end_idx = last_idx - min_lag + 1
            start_idx = max(0, end_idx - window)
            if end_idx > start_idx:
                feature_values[fname] = np.mean(close_vals[start_idx:end_idx])
            else:
                feature_values[fname] = close_vals[last_idx - min_lag]

    # EMA dari Close (shifted by min_lag)
    for span, fname in [(7, "EMA7_Close"), (14, "EMA14_Close")]:
        if fname in feature_names:
            end_idx = last_idx - min_lag + 1
            series = pd.Series(close_vals[:end_idx])
            ema_val = series.ewm(span=span, adjust=False).mean().iloc[-1]
            feature_values[fname] = ema_val

    # Volatility dari HL spread (shifted by min_lag)
    hl_spread = df["High"].values - df["Low"].values
    for window, fname in [(7, "Volatility_7d"), (14, "Volatility_14d")]:
        if fname in feature_names:
            end_idx = last_idx - min_lag + 1
            start_idx = max(0, end_idx - window)
            if end_idx - start_idx > 1:
                feature_values[fname] = np.std(hl_spread[start_idx:end_idx], ddof=1)
            else:
                feature_values[fname] = 0

    # HL_Spread lags
    for i in range(min_lag, min_lag + 3):
        fname = f"HL_Spread_lag{i}"
        if fname in feature_names:
            feature_values[fname] = hl_spread[last_idx - i]

    # OC_Spread lags
    oc_spread = df["Close"].values - df["Open"].values
    for i in range(min_lag, min_lag + 3):
        fname = f"OC_Spread_lag{i}"
        if fname in feature_names:
            feature_values[fname] = oc_spread[last_idx - i]

    row = np.array([feature_values.get(f, 0) for f in feature_names]).reshape(1, -1)
    return row
