"""
Evaluasi - Bandingkan prediksi vs data aktual (per bulan)
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from db.engine import get_session
from db.repository import get_model_runs, get_model_run, get_evaluation_for_run

st.set_page_config(page_title="Evaluasi", page_icon="📏", layout="wide")
st.title("Evaluasi Prediksi")

session = get_session()
runs = get_model_runs(session, limit=50)

# Run yang punya evaluasi: mode forecast_and_evaluate atau backtest dengan MAPE
eval_runs = [
    r for r in runs
    if r.get("eval_mape") is not None
    or r["mode"] in ("forecast_and_evaluate", "backtest")
]

if not eval_runs:
    st.info(
        "Belum ada evaluasi. Jalankan prediksi mode **Forecast + Evaluasi** atau "
        "**Backtest** di halaman **Prediksi Baru**."
    )
    session.close()
    st.stop()

# Ringkasan
st.subheader("Ringkasan Evaluasi")
summary = pd.DataFrame(eval_runs)
summary["run_at"] = pd.to_datetime(summary["run_at"]).dt.strftime("%Y-%m-%d %H:%M")
summary["eval_mape"] = summary["eval_mape"].apply(
    lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
)
summary["eval_rmse"] = summary["eval_rmse"].apply(
    lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "-"
)
summary["eval_ci_coverage"] = summary["eval_ci_coverage"].apply(
    lambda x: f"{x:.1f}%" if pd.notnull(x) else "-"
)
display_cols = [
    "id", "run_at", "mode", "train_end_date",
    "eval_mape", "eval_rmse", "eval_ci_coverage", "status",
]
st.dataframe(summary[display_cols], use_container_width=True, hide_index=True)

st.divider()
st.subheader("Detail Evaluasi")

eval_ids = [r["id"] for r in eval_runs]
selected_id = st.selectbox(
    "Pilih Run ID", eval_ids, format_func=lambda x: f"Run #{x}"
)

if selected_id:
    run = get_model_run(session, selected_id)
    eval_df = get_evaluation_for_run(session, selected_id)

    if eval_df.empty:
        st.warning("Tidak ada data evaluasi untuk run ini.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "MAPE",
            f"{run['eval_mape']:.2f}%" if run.get("eval_mape") else "-",
        )
        c2.metric(
            "RMSE",
            f"Rp {run['eval_rmse']:,.0f}" if run.get("eval_rmse") else "-",
        )
        c3.metric(
            "CI Coverage",
            f"{run['eval_ci_coverage']:.1f}%"
            if run.get("eval_ci_coverage")
            else "-",
        )

        eval_df = eval_df.copy()
        eval_df["Target_Month"] = pd.to_datetime(eval_df["Target_Month"])
        months_str = eval_df["Target_Month"].dt.strftime("%Y-%m")
        error_pct = eval_df["Error (%)"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        signed_err = eval_df["Predicted_Low"] - eval_df["Actual_Low"]
        colors = ["#4CAF50" if e >= 0 else "#F44336" for e in signed_err]
        ax1.bar(months_str, error_pct, color=colors, alpha=0.7)
        ax1.axhline(
            y=pd.Series(error_pct).mean(), color="#FF9800", linestyle="--",
            label=f"Mean: {pd.Series(error_pct).mean():.2f}%",
        )
        ax1.set_title("Error per Bulan (%)")
        ax1.set_ylabel("Error (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.tick_params(axis="x", rotation=45)

        ax2.plot(
            months_str, eval_df["Actual_Low"] / 1e9,
            label="Aktual", color="#4CAF50", linewidth=2, marker="o",
        )
        ax2.plot(
            months_str, eval_df["Predicted_Low"] / 1e9,
            label="Prediksi", color="#E91E63", linewidth=2, marker="s",
            linestyle="--",
        )
        ax2.set_title("Aktual vs Prediksi")
        ax2.set_ylabel("Harga Low (Miliar IDR)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Tabel Perbandingan")
        display = eval_df.copy()
        display["Target_Month"] = display["Target_Month"].dt.strftime("%Y-%m")
        display["Actual_Low"] = display["Actual_Low"].apply(lambda x: f"Rp {x:,.0f}")
        display["Predicted_Low"] = display["Predicted_Low"].apply(
            lambda x: f"Rp {x:,.0f}"
        )
        display["Error (Rp)"] = display["Error (Rp)"].apply(
            lambda x: f"Rp {x:,.0f}"
        )
        display["Error (%)"] = display["Error (%)"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display, use_container_width=True, hide_index=True)

session.close()
