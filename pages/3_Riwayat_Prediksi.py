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
from src.ardl_ecm import explain

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

runs_df["backtest_span"] = [
    f"{int(m)} bln" if str(mode).lower() == "backtest" and pd.notna(m) else "-"
    for mode, m in zip(runs_df["mode"], runs_df["backtest_months"])
]

display_cols = [
    "id", "run_at", "mode", "backtest_span", "train_end_date", "endog_lag",
    "exog_orders", "bounds_f_stat", "cointegration", "eval_mape", "status",
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
        "backtest_span": st.column_config.TextColumn("Backtest (bulan)"),
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

    # ====================================================================
    # Rincian tahapan pipeline (ringkas, dari kolom DB yang tersimpan)
    # ====================================================================
    st.divider()
    st.subheader("Rincian Tahapan Pipeline (ringkas)")
    st.caption(
        "Narasi metodologi tiap tahap + angka ringkasan yang tersimpan di database. "
        "Statistik detail (tabel ADF, kandidat lag, koefisien, diagnostik residual, "
        "proyeksi VAR) hanya tersedia di halaman **🔮 Prediksi Baru** saat run dijalankan."
    )

    _detail_na = "Detail hanya tersedia di halaman Prediksi Baru saat run dijalankan."

    # Tahap 1-3: data dari price_history + resample
    price_df_h = get_price_history(session)
    with st.expander("🔽 Tahap 1–3 — Data, Resampling & Transformasi"):
        st.markdown(explain.NARASI[1])
        st.markdown(explain.NARASI[2])
        st.markdown(explain.NARASI[3])
        if not price_df_h.empty:
            st.write(
                f"Riwayat harga harian di DB: **{len(price_df_h)}** baris · "
                f"{price_df_h['Time'].min().date()} s/d {price_df_h['Time'].max().date()}"
            )
            mh = resample_to_monthly(price_df_h.set_index("Time"), drop_partial=False)
            st.write(f"Setara **{len(mh)}** baris bulanan (drop_partial=False).")
        st.write(
            f"Log-transform run ini: **{'Ya' if run['log_transform'] else 'Tidak'}**"
        )

    with st.expander("🔽 Tahap 4 — Uji Stasioneritas (ADF)"):
        st.markdown(explain.NARASI[4])
        st.info(_detail_na)

    with st.expander("🔽 Tahap 5 — Pemilihan Lag & Estimasi ARDL"):
        st.markdown(explain.NARASI[5])
        st.write(
            f"Lag endogen terpilih (p): **{run.get('endog_lag')}** · "
            f"Order eksogen: **{run.get('exog_orders')}** · IC: **{run['ic'].upper()}**"
        )
        if run.get("train_r2") is not None:
            st.write(f"R² training: **{run['train_r2']:.4f}**")
        st.info(f"Tabel kandidat IC & koefisien penuh: {_detail_na.lower()}")

    with st.expander("🔽 Tahap 6 — Uji Kointegrasi (Bounds Test)"):
        st.markdown(explain.NARASI[6])
        if run.get("bounds_f_stat") is not None:
            st.write(
                f"F-statistik: **{run['bounds_f_stat']:.4f}** · "
                f"Verdict: **{run.get('cointegration')}**"
            )
        st.info(f"Tabel batas kritis lengkap: {_detail_na.lower()}")

    with st.expander("🔽 Tahap 7 — Diagnostik ECM"):
        st.markdown(explain.NARASI[7])
        lam = run.get("lambda_ecm")
        hl = run.get("half_life_months")
        if lam is not None:
            st.write(f"λ (speed of adjustment): **{lam:.6f}**")
        if hl is not None:
            st.write(f"Half-life: **{hl:.2f}** bulan")
        st.info(f"Koefisien jangka panjang & diagnostik residual: {_detail_na.lower()}")

    with st.expander("🔽 Tahap 8–9 — Proyeksi VAR & Forecasting"):
        st.markdown(explain.NARASI[8])
        st.markdown(explain.NARASI[9])
        st.info(f"Detail proyeksi VAR & dekomposisi forecast: {_detail_na.lower()} "
                "Hasil prediksi akhir tersedia pada tabel di atas.")

session.close()
