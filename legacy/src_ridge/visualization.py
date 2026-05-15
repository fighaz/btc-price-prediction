"""
Visualisasi chart untuk forecast dan evaluasi
"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta


def create_forecast_chart(df, forecast_df, merged_eval=None):
    """
    Chart forecast dengan confidence band dan optional evaluasi overlay.

    Returns:
        matplotlib Figure object
    """
    has_eval = merged_eval is not None and len(merged_eval) > 0

    n_plots = 3 if has_eval else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))

    # Chart 1: Historis + Forecast + Confidence Band
    ax1 = axes[0]

    context_start = df["Time"].max() - timedelta(days=60)
    recent = df[df["Time"] >= context_start]

    ax1.plot(recent["Time"], recent["Low"] / 1e9,
             label="Data Historis (Low)", color="#2196F3", linewidth=1.5)
    ax1.plot(forecast_df["Time"], forecast_df["Predicted_Low"] / 1e9,
             label="Forecast ARDL", color="#E91E63", linewidth=2, linestyle="--")
    ax1.fill_between(forecast_df["Time"],
                     forecast_df["Confidence_Lower"] / 1e9,
                     forecast_df["Confidence_Upper"] / 1e9,
                     alpha=0.2, color="#E91E63", label="95% Confidence Interval")
    ax1.axvline(x=df["Time"].max(), color="gray", linestyle="--",
                linewidth=1.5, label="Batas Data Historis")

    if has_eval:
        ax1.plot(pd.to_datetime(merged_eval["Date"]),
                 merged_eval["Actual_Low"] / 1e9,
                 label="Aktual (Evaluasi)", color="#4CAF50", linewidth=1.5,
                 marker="o", markersize=3)

    ax1.set_title("Recursive Forecast Harga Low BTC/IDR - ARDL Model",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Tanggal", fontsize=11)
    ax1.set_ylabel("Harga Low (Miliar IDR)", fontsize=11)
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Chart 2: Forecast saja (zoom in)
    ax2 = axes[1]

    ax2.plot(forecast_df["Time"], forecast_df["Predicted_Low"] / 1e9,
             label="Forecast", color="#E91E63", linewidth=2, marker="o", markersize=4)
    ax2.fill_between(forecast_df["Time"],
                     forecast_df["Confidence_Lower"] / 1e9,
                     forecast_df["Confidence_Upper"] / 1e9,
                     alpha=0.2, color="#E91E63", label="95% CI")

    if has_eval:
        ax2.plot(pd.to_datetime(merged_eval["Date"]),
                 merged_eval["Actual_Low"] / 1e9,
                 label="Aktual", color="#4CAF50", linewidth=2,
                 marker="s", markersize=4)

    ax2.set_title("Detail Forecast (Zoom)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Tanggal", fontsize=11)
    ax2.set_ylabel("Harga Low (Miliar IDR)", fontsize=11)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", rotation=45)

    # Chart 3: Error distribution (jika ada evaluasi)
    if has_eval:
        ax3 = axes[2]
        error_pct = ((merged_eval["Predicted_Low"] - merged_eval["Actual_Low"])
                     / merged_eval["Actual_Low"] * 100)

        colors = ["#4CAF50" if e >= 0 else "#F44336" for e in error_pct]
        ax3.bar(range(len(error_pct)), error_pct, color=colors, alpha=0.7, edgecolor="white")
        ax3.axhline(y=0, color="black", linewidth=0.5)
        ax3.axhline(y=error_pct.mean(), color="#FF9800", linestyle="--",
                    linewidth=2, label=f"Mean: {error_pct.mean():.2f}%")

        ax3.set_title("Prediction Error per Hari (%)", fontsize=14, fontweight="bold")
        ax3.set_xlabel("Hari ke-", fontsize=11)
        ax3.set_ylabel("Error (%)", fontsize=11)
        ax3.set_xticks(range(0, len(error_pct), max(1, len(error_pct) // 10)))
        ax3.legend(loc="best", fontsize=10)
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def create_test_chart(test_df, full_df):
    """
    Chart perbandingan actual vs predicted untuk mode train/test split.

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # Chart 1: Actual vs Predicted (Test Period)
    ax1 = axes[0]
    ax1.plot(test_df["Time"], test_df["Low"] / 1e9,
             label="Actual Low", color="#2196F3", linewidth=1.5)
    ax1.plot(test_df["Time"], test_df["Pred_ARDL"] / 1e9,
             label="ARDL Prediction", color="#E91E63", linewidth=1.5)
    ax1.fill_between(test_df["Time"],
                     test_df["Low"] / 1e9,
                     test_df["Pred_ARDL"] / 1e9,
                     alpha=0.2, color="gray", label="Prediction Gap")

    test_start = test_df["Time"].min().date()
    test_end = test_df["Time"].max().date()
    ax1.set_title(f"Prediksi Harga Low BTC/IDR - ARDL Model\nTest Period: {test_start} s/d {test_end}",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Tanggal", fontsize=11)
    ax1.set_ylabel("Harga Low (Miliar IDR)", fontsize=11)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Chart 2: Error Distribution
    ax2 = axes[1]
    error_pct = (test_df["Pred_ARDL"] - test_df["Low"]) / test_df["Low"] * 100
    ax2.hist(error_pct, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax2.axvline(x=error_pct.mean(), color="#FF9800", linestyle="--",
                linewidth=2, label=f"Mean Error: {error_pct.mean():.2f}%")
    ax2.axvline(x=error_pct.median(), color="#4CAF50", linestyle="--",
                linewidth=2, label=f"Median Error: {error_pct.median():.2f}%")
    ax2.set_title("Distribusi Prediction Error (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Error (%)", fontsize=11)
    ax2.set_ylabel("Frekuensi", fontsize=11)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Chart 3: Historical context
    ax3 = axes[2]
    context_start = test_df["Time"].min() - timedelta(days=60)
    recent_df = full_df[full_df["Time"] >= context_start]
    ax3.plot(recent_df["Time"], recent_df["Low"] / 1e9,
             label="Historical Low", color="#2196F3", linewidth=1.5, alpha=0.7)
    ax3.plot(test_df["Time"], test_df["Pred_ARDL"] / 1e9,
             label="ARDL Prediction (Test Period)", color="#E91E63", linewidth=1.5)
    ax3.axvline(x=test_df["Time"].iloc[0], color="gray", linestyle="--",
                linewidth=1.5, label="Training/Test Split")
    ax3.set_title("Konteks Historis dengan Prediksi ARDL", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Tanggal", fontsize=11)
    ax3.set_ylabel("Harga Low (Miliar IDR)", fontsize=11)
    ax3.legend(loc="best", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig
