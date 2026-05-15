"""
Optimasi parameter ARDL menggunakan AIC
"""
import logging
import numpy as np
from sklearn.linear_model import LinearRegression

from src.features import prepare_ardl_features, get_ardl_features

logger = logging.getLogger(__name__)


def select_optimal_lags(df, max_p=10, max_q=10, min_lag=2, progress_callback=None):
    """
    Select optimal lag order menggunakan AIC (Akaike Information Criterion).
    Grid search step=1.

    Parameters:
        df: DataFrame data historis
        max_p, max_q: batas maksimum lag
        min_lag: minimum lag offset
        progress_callback: optional callable(current, total) untuk progress bar

    Returns:
        (best_p, best_q, all_results)
    """
    logger.info(f"Mencari optimal lag order (grid search step=1, min_lag={min_lag})...")

    best_aic = np.inf
    best_p = 1
    best_q = 1
    results = []
    total = max_p * max_q
    current = 0

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            current += 1
            if progress_callback:
                progress_callback(current, total)
            try:
                temp_df = prepare_ardl_features(df.copy(), p_lags=p, q_lags=q, min_lag=min_lag)
                if len(temp_df) < 100:
                    continue

                train_size = int(len(temp_df) * 0.8)
                train_df = temp_df.iloc[:train_size]

                features = get_ardl_features(p, q, min_lag=min_lag)
                available_features = [f for f in features if f in train_df.columns]

                X = train_df[available_features].values
                y = train_df["Low"].values

                model = LinearRegression()
                model.fit(X, y)

                y_pred = model.predict(X)
                n = len(y)
                k = X.shape[1] + 1
                rss = np.sum((y - y_pred) ** 2)
                aic = n * np.log(rss / n) + 2 * k

                results.append({"p": p, "q": q, "aic": aic})

                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    best_q = q
            except Exception:
                continue

    logger.info(f"Optimal lags: p={best_p}, q={best_q}, AIC={best_aic:.2f}")
    return best_p, best_q, results
