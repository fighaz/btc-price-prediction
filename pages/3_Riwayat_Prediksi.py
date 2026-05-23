"""
Riwayat Prediksi - Browse histori model runs + config dari database
"""
import altair as alt
import pandas as pd
import streamlit as st

from db.engine import get_session
from db.repository import (
    get_model_runs,
    get_model_run,
    get_predictions_for_run,
    get_price_history,
)
from src.ardl_ecm.data import resample_to_monthly
from src.ardl_ecm.charts import forecast_with_history_chart

st.set_page_config(page_title="Riwayat Prediksi", page_icon="📋", layout="wide")
st.title("📋 Riwayat Prediksi")
st.caption("Telusuri seluruh histori run beserta konfigurasinya.")

session = get_session()
runs = get_model_runs(session, limit=50)

if not runs:
    st.info(
        "Belum ada riwayat prediksi. Buka halaman **🔮 Prediksi Baru** untuk "
        "menjalankan prediksi pertama."
    )
    session.close()
    st.stop()

# ============================================================================
# Tabel model runs
# ============================================================================
st.subheader("Daftar Model Runs")
runs_df = pd.DataFrame(runs)
runs_df["run_at"] = pd.to_datetime(runs_df["run_at"])

display_cols = [
    "id", "run_at", "mode", "train_end_date", "endog_lag", "exog_orders",
    "bounds_f_stat", "cointegration", "eval_mape", "status",
]
st.dataframe(
    runs_df[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "id": st.column_config.NumberColumn("Run ID", format="%d"),
        "run_at": st.column_config.DatetimeColumn(
            "Waktu Run", format="YYYY-MM-DD HH:mm"
        ),
        "mode": st.column_config.TextColumn("Mode"),
        "train_end_date": st.column_config.DateColumn("Train End"),
        "endog_lag": st.column_config.NumberColumn("Lag p", format="%d"),
        "exog_orders": st.column_config.TextColumn("Exog Orders"),
        "bounds_f_stat": st.column_config.NumberColumn(
            "Bounds F", format="%.4f"
        ),
        "cointegration": st.column_config.TextColumn("Kointegrasi"),
        "eval_mape": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
        "status": st.column_config.TextColumn("Status"),
    },
)

# ============================================================================
# Detail per run
# ============================================================================
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
    cfg_cols[0].metric("Mode", run["mode"])
    cfg_cols[1].metric("Train End", str(run["train_end_date"]))
    cfg_cols[2].metric("Log Transform", "Ya" if run["log_transform"] else "Tidak")
    cfg_cols[3].metric(
        "Rolling Window",
        f"{run['rolling_window_years']} thn"
        if run["rolling_window_years"]
        else "Semua data",
    )
    cfg_cols2 = st.columns(4)
    cfg_cols2[0].metric("Max Lag Endog", run["max_lag_endog"])
    cfg_cols2[1].metric("Max Lag Exog", run["max_lag_exog"])
    cfg_cols2[2].metric("IC", run["ic"].upper())
    cfg_cols2[3].metric(
        "Backtest Months",
        run["backtest_months"] if run["backtest_months"] else "-",
    )

    pred_df = get_predictions_for_run(session, selected_id)
    if pred_df.empty:
        st.warning("Tidak ada data prediksi untuk run ini.")
    else:
        pred_df = pred_df.copy()
        pred_df["Target_Month"] = pd.to_datetime(pred_df["Target_Month"])

        # History dari DB untuk konteks chart
        price_df = get_price_history(session)
        monthly_history = pd.DataFrame()
        if not price_df.empty:
            monthly_history = resample_to_monthly(
                price_df.set_index("Time"), drop_partial=False
            )

        st.markdown("**Chart Prediksi**")
        if not monthly_history.empty:
            fc_input = pred_df.set_index("Target_Month")[
                ["Predicted_Low", "CI_Lower", "CI_Upper"]
            ]
            st.altair_chart(
                forecast_with_history_chart(monthly_history, fc_input),
                use_container_width=True,
            )
        else:
            # Fallback: hanya prediksi, tanpa history
            chart_df = pd.DataFrame(
                {
                    "Bulan": pred_df["Target_Month"].dt.strftime("%Y-%m"),
                    "Prediksi": pred_df["Predicted_Low"] / 1e9,
                }
            )
            st.altair_chart(
                alt.Chart(chart_df)
                .mark_line(point=True, color="#E91E63")
                .encode(
                    x=alt.X("Bulan:O", title="Bulan"),
                    y=alt.Y("Prediksi:Q", title="Prediksi Low (Miliar IDR)"),
                    tooltip=["Bulan", alt.Tooltip("Prediksi:Q", format=",.3f")],
                )
                .properties(height=380)
                .interactive(),
                use_container_width=True,
            )

        st.markdown("**Tabel Prediksi**")
        display = pred_df.copy()
        display["Target_Month"] = display["Target_Month"].dt.strftime("%Y-%m")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Target_Month": st.column_config.TextColumn("Bulan Target"),
                "Predicted_Low": st.column_config.NumberColumn(
                    "Prediksi Low", format="Rp %.0f"
                ),
                "CI_Lower": st.column_config.NumberColumn(
                    "CI Bawah", format="Rp %.0f"
                ),
                "CI_Upper": st.column_config.NumberColumn(
                    "CI Atas", format="Rp %.0f"
                ),
            },
        )

        csv = pred_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            file_name=f"prediction_run_{selected_id}.csv",
            mime="text/csv",
        )

session.close()
