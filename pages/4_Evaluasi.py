"""
Evaluasi - Bandingkan prediksi vs data aktual (per bulan)
"""
import pandas as pd
import streamlit as st

from db.engine import get_session
from db.repository import get_model_runs, get_model_run, get_evaluation_for_run
from src.ardl_ecm.charts import error_bar_chart, actual_vs_predicted_chart

st.set_page_config(page_title="Evaluasi", page_icon="📏", layout="wide")
st.title("📏 Evaluasi Prediksi")
st.caption("Bandingkan akurasi prediksi terhadap harga aktual.")

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
        "**Backtest** di halaman **🔮 Prediksi Baru**."
    )
    session.close()
    st.stop()

# ============================================================================
# Ringkasan
# ============================================================================
st.subheader("Ringkasan Evaluasi")
summary = pd.DataFrame(eval_runs)
summary["run_at"] = pd.to_datetime(summary["run_at"])
display_cols = [
    "id", "run_at", "mode", "train_end_date",
    "eval_mape", "eval_mae", "eval_rmse", "eval_r2", "eval_ci_coverage", "status",
]
st.dataframe(
    summary[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "id": st.column_config.NumberColumn("Run ID", format="%d"),
        "run_at": st.column_config.DatetimeColumn(
            "Waktu Run", format="YYYY-MM-DD HH:mm"
        ),
        "mode": st.column_config.TextColumn("Mode"),
        "train_end_date": st.column_config.DateColumn("Train End"),
        "eval_mape": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
        "eval_mae": st.column_config.NumberColumn("MAE", format="Rp %.0f"),
        "eval_rmse": st.column_config.NumberColumn("RMSE", format="Rp %.0f"),
        "eval_r2": st.column_config.NumberColumn("R²", format="%.4f"),
        "eval_ci_coverage": st.column_config.NumberColumn(
            "CI Coverage", format="%.1f%%"
        ),
        "status": st.column_config.TextColumn("Status"),
    },
)

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
        # Kartu metrik
        c1, c2, c3, c4, c5 = st.columns(5)
        mape = run.get("eval_mape")
        c1.metric(
            "MAPE", f"{mape:.2f}%" if mape is not None else "-",
            delta="Target < 10%" if mape is not None else None,
            delta_color="off",
        )
        mae = run.get("eval_mae")
        c2.metric(
            "MAE", f"Rp {mae/1e6:,.1f} Jt" if mae is not None else "-"
        )
        rmse = run.get("eval_rmse")
        c3.metric(
            "RMSE", f"Rp {rmse/1e6:,.1f} Jt" if rmse is not None else "-"
        )
        r2 = run.get("eval_r2")
        c4.metric(
            "R²", f"{r2:.4f}" if r2 is not None else "-",
            delta="Target > 0.9" if r2 is not None else None,
            delta_color="off",
        )
        ci = run.get("eval_ci_coverage")
        c5.metric(
            "CI Coverage", f"{ci:.1f}%" if ci is not None else "-",
            delta="Target ~95%" if ci is not None else None,
            delta_color="off",
        )

        eval_df = eval_df.copy()
        eval_df["Target_Month"] = pd.to_datetime(eval_df["Target_Month"])

        st.markdown("**Aktual vs Prediksi**")
        st.altair_chart(
            actual_vs_predicted_chart(eval_df), use_container_width=True
        )

        st.markdown("**Error per Bulan**")
        st.altair_chart(error_bar_chart(eval_df), use_container_width=True)
        st.caption(
            "Hijau = prediksi lebih tinggi dari aktual (over) · "
            "merah = prediksi lebih rendah (under)."
        )

        st.markdown("**Tabel Perbandingan**")
        display = eval_df.copy()
        display["Target_Month"] = display["Target_Month"].dt.strftime("%Y-%m")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Target_Month": st.column_config.TextColumn("Bulan"),
                "Actual_Low": st.column_config.NumberColumn(
                    "Aktual Low", format="Rp %.0f"
                ),
                "Predicted_Low": st.column_config.NumberColumn(
                    "Prediksi Low", format="Rp %.0f"
                ),
                "Error (Rp)": st.column_config.NumberColumn(
                    "Error (Rp)", format="Rp %.0f"
                ),
                "Error (%)": st.column_config.NumberColumn(
                    "Error (%)", format="%.2f%%"
                ),
                "Dalam CI": st.column_config.TextColumn("Dalam CI"),
            },
        )

session.close()
