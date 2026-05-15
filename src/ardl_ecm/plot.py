"""
Visualisasi untuk ARDL-ECM monthly.

Semua fungsi mengembalikan matplotlib Figure (bukan savefig) supaya bisa
dipakai langsung oleh st.pyplot().
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_single_forecast(forecast_df, monthly_history, eval_result=None):
    """
    Chart forecast 1 bulan: history + forecast + optional aktual.

    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    recent = monthly_history.tail(36)
    months_hist = [m.strftime("%Y-%m") for m in recent.index]
    fc_label = forecast_df.index[0].strftime("%Y-%m")

    ax.plot(
        months_hist, recent["Low"] / 1e9, "o-", color="#2196F3",
        linewidth=1.5, markersize=5, label="Historis Monthly Low",
    )
    fc_val = forecast_df.iloc[0]["Predicted_Low"] / 1e9
    ax.errorbar(
        [fc_label], [fc_val],
        yerr=[
            [fc_val - forecast_df.iloc[0]["CI_Lower"] / 1e9],
            [forecast_df.iloc[0]["CI_Upper"] / 1e9 - fc_val],
        ],
        fmt="*", color="#E91E63", markersize=18, capsize=8,
        label=f"Forecast {fc_label}",
    )
    if eval_result:
        ax.plot(
            [fc_label], [eval_result["actual_low"] / 1e9], "D",
            color="#4CAF50", markersize=12,
            label=f"Aktual {fc_label}: Rp {eval_result['actual_low']:,.0f}",
        )

    ax.set_title(
        f"Monthly ARDL-ECM Forecast - {fc_label}", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Low (Miliar IDR)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_backtest(results_df, metrics):
    """
    Chart 4-panel hasil rolling backtest.

    Returns matplotlib Figure.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 18))
    valid = results_df.dropna(subset=["predicted_low"]).copy()
    months_str = [m.strftime("%Y-%m") for m in valid["month"]]

    # Panel 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(
        months_str, valid["actual_low"] / 1e9, "o-", color="#4CAF50",
        linewidth=2, markersize=7, label="Aktual",
    )
    ax1.plot(
        months_str, valid["predicted_low"] / 1e9, "s--", color="#E91E63",
        linewidth=2, markersize=7, label="Prediksi",
    )
    ax1.fill_between(
        months_str, valid["ci_lower"] / 1e9, valid["ci_upper"] / 1e9,
        color="#E91E63", alpha=0.15, label="95% CI",
    )
    ax1.set_title(
        f"Rolling Backtest - Actual vs Predicted Monthly Low (n={metrics['n']})",
        fontsize=13, fontweight="bold",
    )
    ax1.set_ylabel("Low (Miliar IDR)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=45)

    # Panel 2: Error % per bulan
    ax2 = axes[1]
    err = (
        (valid["predicted_low"] - valid["actual_low"]) / valid["actual_low"] * 100
    ).values
    colors = ["#4CAF50" if e >= 0 else "#F44336" for e in err]
    ax2.bar(months_str, err, color=colors, alpha=0.75, edgecolor="white")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.axhline(
        y=err.mean(), color="#FF9800", linestyle="--", linewidth=2,
        label=f"Mean error: {err.mean():.2f}%",
    )
    ax2.set_title(
        f"Error Bulanan (MAPE rata-rata: {metrics['mape']:.2f}%)",
        fontsize=13, fontweight="bold",
    )
    ax2.set_ylabel("Error (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="x", rotation=45)

    # Panel 3: CI coverage map
    ax3 = axes[2]
    in_ci = valid["in_ci"].astype(int).values
    colors_ci = ["#4CAF50" if x else "#F44336" for x in in_ci]
    ax3.bar(months_str, [1] * len(months_str), color=colors_ci, alpha=0.85,
            edgecolor="white")
    ax3.set_yticks([])
    ax3.set_title(
        f"CI Coverage: {metrics['ci_coverage']:.1f}% "
        f"({valid['in_ci'].sum()}/{len(valid)} dalam 95% CI)  "
        f"[hijau=in CI, merah=miss]",
        fontsize=13, fontweight="bold",
    )
    ax3.tick_params(axis="x", rotation=45)

    # Panel 4: Bounds F-stat per iterasi
    ax4 = axes[3]
    f_vals = valid["bounds_F"].values
    ax4.plot(months_str, f_vals, "o-", color="#1976D2", linewidth=2, markersize=7)
    ax4.axhline(
        y=4.0, color="#F44336", linestyle="--", linewidth=1.5,
        label="approx I(1) upper @5% (3 regressors)",
    )
    ax4.set_title(
        f"Bounds F-stat per Iterasi  ({metrics['coint_rate']:.0f}% iter cointegrated)",
        fontsize=13, fontweight="bold",
    )
    ax4.set_ylabel("F-statistic")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig
