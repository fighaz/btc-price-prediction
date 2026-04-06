"""
Evaluasi - Bandingkan prediksi vs data aktual
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from db.engine import get_session
from db.repository import get_model_runs, get_evaluation_for_run

st.set_page_config(page_title="Evaluasi", page_icon="📏", layout="wide")
st.title("Evaluasi Prediksi")

session = get_session()
runs = get_model_runs(session, limit=50)

# Filter runs yang punya evaluasi
eval_runs = [r for r in runs if r.get("eval_mape") is not None]

if not eval_runs:
    st.info("Belum ada evaluasi. Jalankan prediksi dengan opsi evaluasi di halaman **Prediksi Baru**.")
    session.close()
    st.stop()

# Ringkasan evaluasi
st.subheader("Ringkasan Evaluasi")
eval_summary = pd.DataFrame(eval_runs)
eval_summary["run_at"] = pd.to_datetime(eval_summary["run_at"]).dt.strftime("%Y-%m-%d %H:%M")
eval_summary["eval_mape"] = eval_summary["eval_mape"].apply(lambda x: f"{x:.2f}%" if x else "-")
eval_summary["eval_rmse"] = eval_summary["eval_rmse"].apply(lambda x: f"Rp {x:,.0f}" if x else "-")
eval_summary["eval_ci_coverage"] = eval_summary["eval_ci_coverage"].apply(lambda x: f"{x:.1f}%" if x else "-")

display_cols = ["id", "run_at", "train_end_date", "forecast_days",
                "eval_mape", "eval_rmse", "eval_ci_coverage", "status"]
st.dataframe(eval_summary[display_cols], use_container_width=True, hide_index=True)

# Detail evaluasi
st.divider()
st.subheader("Detail Evaluasi")

eval_ids = [r["id"] for r in eval_runs]
selected_id = st.selectbox("Pilih Run ID", eval_ids, format_func=lambda x: f"Run #{x}")

if selected_id:
    eval_df = get_evaluation_for_run(session, selected_id)

    if eval_df.empty:
        st.warning("Tidak ada data evaluasi untuk run ini.")
    else:
        # Metrics
        run = next(r for r in eval_runs if r["id"] == selected_id)
        col1, col2, col3 = st.columns(3)
        with col1:
            mape = run.get("eval_mape")
            st.metric("MAPE", f"{mape:.2f}%" if mape else "-")
        with col2:
            rmse = run.get("eval_rmse")
            st.metric("RMSE", f"Rp {rmse:,.0f}" if rmse else "-")
        with col3:
            ci = run.get("eval_ci_coverage")
            st.metric("CI Coverage", f"{ci:.1f}%" if ci else "-")

        # Error distribution chart
        eval_df["Tanggal_dt"] = pd.to_datetime(eval_df["Tanggal"])
        error_pct = eval_df["Error (%)"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart error per hari
        colors = ["#4CAF50" if e >= 0 else "#F44336" for e in (eval_df["Predicted_Low"] - eval_df["Actual_Low"])]
        ax1.bar(range(len(error_pct)), error_pct, color=colors, alpha=0.7)
        ax1.axhline(y=pd.Series(error_pct).mean(), color="#FF9800", linestyle="--",
                     label=f"Mean: {pd.Series(error_pct).mean():.2f}%")
        ax1.set_title("Error per Hari (%)")
        ax1.set_xlabel("Hari ke-")
        ax1.set_ylabel("Error (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Actual vs Predicted
        ax2.plot(eval_df["Tanggal_dt"], eval_df["Actual_Low"] / 1e9,
                 label="Aktual", color="#4CAF50", linewidth=2)
        ax2.plot(eval_df["Tanggal_dt"], eval_df["Predicted_Low"] / 1e9,
                 label="Prediksi", color="#E91E63", linewidth=2, linestyle="--")
        ax2.set_title("Aktual vs Prediksi")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Harga Low (Miliar IDR)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tabel detail
        st.subheader("Tabel Perbandingan")
        display = eval_df.copy()
        display["Actual_Low"] = display["Actual_Low"].apply(lambda x: f"Rp {x:,.0f}")
        display["Predicted_Low"] = display["Predicted_Low"].apply(lambda x: f"Rp {x:,.0f}")
        display["Error (Rp)"] = display["Error (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
        display["Error (%)"] = display["Error (%)"].apply(lambda x: f"{x:.2f}%")
        display = display.drop(columns=["Tanggal_dt"], errors="ignore")
        st.dataframe(display, use_container_width=True, hide_index=True)

session.close()
