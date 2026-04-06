"""
ARDL Model: training dan recursive forecasting
"""
import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

from config import ALPHA_CANDIDATES, CV_SPLITS
from src.features import prepare_ardl_features, get_ardl_features, compute_single_row_features

logger = logging.getLogger(__name__)


class ARDLModel:
    """ARDL Model dengan Ridge regularization"""

    def __init__(self, p_lags=7, q_lags=7, regularization="ridge", alpha=1.0):
        self.p_lags = p_lags
        self.q_lags = q_lags
        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.residual_std = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)

        if hasattr(self.model, "coef_"):
            self.feature_importance = np.abs(self.model.coef_)

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


def train_ardl_model(df, p_lags=7, q_lags=7, min_lag=2):
    """
    Training ARDL model pada SELURUH data (untuk production forecasting).

    Returns:
        (ardl_model, df_features, available_features, train_info)
    """
    df_features = prepare_ardl_features(df.copy(), p_lags, q_lags, min_lag=min_lag)

    features = get_ardl_features(p_lags, q_lags, min_lag=min_lag)
    available_features = [f for f in features if f in df_features.columns]

    X_train = df_features[available_features].values
    y_train = df_features["Low"].values

    logger.info(f"Training ARDL (data: {len(df_features)} samples, features: {len(available_features)})")

    # CV untuk alpha
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    best_alpha = 1.0
    best_cv_score = -np.inf

    for alpha in ALPHA_CANDIDATES:
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

    logger.info(f"Best alpha (via CV): {best_alpha}")

    # Train final model
    ardl_model = ARDLModel(p_lags=p_lags, q_lags=q_lags, regularization="ridge", alpha=best_alpha)
    ardl_model.fit(X_train, y_train, available_features)

    train_r2 = r2_score(y_train, ardl_model.predict(X_train))

    train_info = {
        "n_samples": len(df_features),
        "n_features": len(available_features),
        "best_alpha": best_alpha,
        "train_r2": train_r2,
        "residual_std": ardl_model.residual_std,
        "date_start": str(df_features["Time"].min().date()),
        "date_end": str(df_features["Time"].max().date()),
        "top_features": ardl_model.get_top_features(10),
    }

    logger.info(f"R-squared (training): {train_r2:.4f}")
    return ardl_model, df_features, available_features, train_info


def forecast_future(ardl_model, df, feature_names, n_days=31,
                    p_lags=9, q_lags=9, min_lag=2):
    """
    Recursive forecasting: prediksi N hari ke depan TANPA data aktual.

    Returns:
        forecast_df: DataFrame [Time, Predicted_Low, Confidence_Lower, Confidence_Upper, Step]
    """
    logger.info(f"Recursive forecast: {n_days} hari ke depan")

    # Rasio proxy dari 30 hari terakhir
    last_30 = df.tail(30)
    high_low_ratio = (last_30["High"] / last_30["Low"]).median()
    close_low_ratio = (last_30["Close"] / last_30["Low"]).median()
    median_volume = last_30["Volume"].median()

    # History buffer
    buffer_size = min_lag + max(p_lags, q_lags) + 35
    history = df[["Time", "Open", "High", "Low", "Close", "Volume"]].tail(buffer_size).copy()
    history = history.reset_index(drop=True)

    last_date = history["Time"].iloc[-1]
    residual_std = ardl_model.residual_std

    predictions = []

    for step in range(1, n_days + 1):
        forecast_date = last_date + timedelta(days=step)

        feature_row = compute_single_row_features(
            history, p_lags, q_lags, min_lag, feature_names
        )

        pred_low = ardl_model.predict(feature_row)[0]

        ci_width = residual_std * np.sqrt(step) * 1.96
        ci_lower = pred_low - ci_width
        ci_upper = pred_low + ci_width

        predictions.append({
            "Time": forecast_date,
            "Predicted_Low": pred_low,
            "Confidence_Lower": ci_lower,
            "Confidence_Upper": ci_upper,
            "Step": step,
        })

        # Proxy OHLCV untuk recursive step
        proxy_open = history["Close"].iloc[-1]
        proxy_high = pred_low * high_low_ratio
        proxy_close = pred_low * close_low_ratio

        new_row = pd.DataFrame([{
            "Time": forecast_date,
            "Open": proxy_open,
            "High": proxy_high,
            "Low": pred_low,
            "Close": proxy_close,
            "Volume": median_volume,
        }])
        history = pd.concat([history, new_row], ignore_index=True)

    forecast_df = pd.DataFrame(predictions)
    logger.info(f"Forecast selesai: {len(forecast_df)} hari")
    return forecast_df
