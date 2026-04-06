"""
Riwayat Prediksi - Browse histori model runs dan prediksi dari database
"""
import streamlit as st
import pandas as pd
from db.engine import get_session
from db.repository import get_model_runs, get_predictions_for_run

st.set_page_config(page_title="Riwayat Prediksi", page_icon="📋", layout="wide")
st.title("Riwayat Prediksi")

session = get_session()
runs = get_model_runs(session, limit=50)

if not runs:
    st.info("Belum ada riwayat prediksi. Buka halaman **Prediksi Baru** untuk menjalankan prediksi pertama.")
    session.close()
    st.stop()

# Tabel model runs
st.subheader("Daftar Model Runs")
runs_df = pd.DataFrame(runs)
runs_df["run_at"] = pd.to_datetime(runs_df["run_at"]).dt.strftime("%Y-%m-%d %H:%M")
runs_df["train_r2"] = runs_df["train_r2"].apply(lambda x: f"{x:.4f}" if x else "-")
runs_df["eval_mape"] = runs_df["eval_mape"].apply(lambda x: f"{x:.2f}%" if x else "-")

display_cols = ["id", "run_at", "mode", "train_end_date", "forecast_days",
                "p_lags", "q_lags", "alpha", "train_r2", "eval_mape", "status"]
st.dataframe(runs_df[display_cols], use_container_width=True, hide_index=True)

# Detail per run
st.divider()
st.subheader("Detail Prediksi")

run_ids = [r["id"] for r in runs]
selected_id = st.selectbox("Pilih Run ID", run_ids, format_func=lambda x: f"Run #{x}")

if selected_id:
    pred_df = get_predictions_for_run(session, selected_id)

    if pred_df.empty:
        st.warning("Tidak ada data prediksi untuk run ini.")
    else:
        # Chart
        pred_df["Tanggal_dt"] = pd.to_datetime(pred_df["Tanggal"])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(pred_df["Tanggal_dt"], pred_df["Predicted_Low"] / 1e9,
                color="#E91E63", linewidth=2, marker="o", markersize=3)
        ax.fill_between(pred_df["Tanggal_dt"],
                        pred_df["Confidence_Lower"] / 1e9,
                        pred_df["Confidence_Upper"] / 1e9,
                        alpha=0.2, color="#E91E63")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga Low (Miliar IDR)")
        ax.set_title(f"Forecast Run #{selected_id}")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Tabel
        display = pred_df.copy()
        display["Predicted_Low"] = display["Predicted_Low"].apply(lambda x: f"Rp {x:,.0f}")
        display["Confidence_Lower"] = display["Confidence_Lower"].apply(lambda x: f"Rp {x:,.0f}")
        display["Confidence_Upper"] = display["Confidence_Upper"].apply(lambda x: f"Rp {x:,.0f}")
        display = display.drop(columns=["Tanggal_dt"], errors="ignore")
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Download CSV
        csv = pred_df.drop(columns=["Tanggal_dt"], errors="ignore").to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"prediction_run_{selected_id}.csv",
            mime="text/csv",
        )

session.close()
