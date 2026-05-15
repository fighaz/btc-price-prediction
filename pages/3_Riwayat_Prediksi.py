"""
Riwayat Prediksi - Browse histori model runs + config dari database
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from db.engine import get_session
from db.repository import get_model_runs, get_model_run, get_predictions_for_run

st.set_page_config(page_title="Riwayat Prediksi", page_icon="📋", layout="wide")
st.title("Riwayat Prediksi")

session = get_session()
runs = get_model_runs(session, limit=50)

if not runs:
    st.info(
        "Belum ada riwayat prediksi. Buka halaman **Prediksi Baru** untuk "
        "menjalankan prediksi pertama."
    )
    session.close()
    st.stop()

# Tabel model runs
st.subheader("Daftar Model Runs")
runs_df = pd.DataFrame(runs)
runs_df["run_at"] = pd.to_datetime(runs_df["run_at"]).dt.strftime("%Y-%m-%d %H:%M")
runs_df["bounds_f_stat"] = runs_df["bounds_f_stat"].apply(
    lambda x: f"{x:.4f}" if pd.notnull(x) else "-"
)
runs_df["eval_mape"] = runs_df["eval_mape"].apply(
    lambda x: f"{x:.2f}%" if pd.notnull(x) else "-"
)

display_cols = [
    "id", "run_at", "mode", "train_end_date", "endog_lag", "exog_orders",
    "bounds_f_stat", "cointegration", "eval_mape", "status",
]
st.dataframe(runs_df[display_cols], use_container_width=True, hide_index=True)

# Detail per run
st.divider()
st.subheader("Detail Run")

run_ids = [r["id"] for r in runs]
selected_id = st.selectbox(
    "Pilih Run ID", run_ids, format_func=lambda x: f"Run #{x}"
)

if selected_id:
    run = get_model_run(session, selected_id)

    # Config run
    st.markdown("**Konfigurasi Run**")
    cfg_cols = st.columns(4)
    cfg_cols[0].write(f"Mode: `{run['mode']}`")
    cfg_cols[0].write(f"Train end: `{run['train_end_date']}`")
    cfg_cols[1].write(f"Log transform: `{run['log_transform']}`")
    cfg_cols[1].write(f"Rolling window: `{run['rolling_window_years']} thn`")
    cfg_cols[2].write(f"Max lag endog: `{run['max_lag_endog']}`")
    cfg_cols[2].write(f"Max lag exog: `{run['max_lag_exog']}`")
    cfg_cols[3].write(f"IC: `{run['ic']}`")
    cfg_cols[3].write(
        f"Backtest months: `{run['backtest_months']}`"
        if run["backtest_months"]
        else "Backtest months: `-`"
    )

    pred_df = get_predictions_for_run(session, selected_id)
    if pred_df.empty:
        st.warning("Tidak ada data prediksi untuk run ini.")
    else:
        pred_df = pred_df.copy()
        pred_df["Target_Month"] = pd.to_datetime(pred_df["Target_Month"])
        months_str = pred_df["Target_Month"].dt.strftime("%Y-%m")

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(
            months_str, pred_df["Predicted_Low"] / 1e9,
            color="#E91E63", linewidth=2, marker="o", markersize=5,
        )
        if pred_df["CI_Lower"].notna().all() and pred_df["CI_Upper"].notna().all():
            ax.fill_between(
                months_str, pred_df["CI_Lower"] / 1e9, pred_df["CI_Upper"] / 1e9,
                alpha=0.2, color="#E91E63",
            )
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Harga Low (Miliar IDR)")
        ax.set_title(f"Prediksi Run #{selected_id}")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        display = pred_df.copy()
        display["Target_Month"] = display["Target_Month"].dt.strftime("%Y-%m")
        for col in ["Predicted_Low", "CI_Lower", "CI_Upper"]:
            display[col] = display[col].apply(
                lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)

        csv = pred_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"prediction_run_{selected_id}.csv",
            mime="text/csv",
        )

session.close()
