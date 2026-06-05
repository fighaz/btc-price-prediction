"""
Prediksi Baru - Jalankan pipeline ARDL-ECM monthly
Mode: Forecast | Forecast + Evaluasi | Backtest
"""
import logging
from datetime import date

import pandas as pd
import streamlit as st

# Nama bulan Indonesia untuk selectbox month picker
_BULAN_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

from db.engine import init_db, get_session
from db.repository import (
    save_model_run,
    update_model_run_result,
    save_monthly_predictions,
    save_backtest_predictions,
    save_single_evaluation,
    save_backtest_evaluations,
    save_price_history,
)
from src.pipeline import (
    run_monthly_forecast,
    run_monthly_forecast_and_evaluate,
    run_monthly_backtest,
)
from src.ardl_ecm.charts import forecast_with_history_chart, backtest_chart
from src.ardl_ecm import explain
from src.ardl_ecm.config import TREND

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Prediksi Baru", page_icon="🔮", layout="wide")
st.title("Prediksi Baru")
st.markdown("Jalankan pipeline ARDL-ECM monthly dengan parameter yang bisa disesuaikan.")

init_db()

MODE_LABELS = {
    "Forecast": "forecast",
    "Forecast + Evaluasi": "forecast_and_evaluate",
    "Backtest (walk-forward)": "backtest",
}

# --- Pilihan mode ---
mode_label = st.radio(
    "Mode prediksi",
    list(MODE_LABELS.keys()),
    horizontal=True,
    help=(
        "Forecast: prediksi Low bulan berjalan — dijalankan di awal bulan, "
        "menggunakan harga Open hari pertama sebagai sinyal tambahan. "
        "Forecast + Evaluasi: prediksi lalu bandingkan dengan aktual. "
        "Backtest: walk-forward N bulan untuk mengukur akurasi historis "
        "(tiap iterasi menginjeksikan Open aktual bulan target)."
    ),
)
mode = MODE_LABELS[mode_label]

# --- Form config ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    today = date.today()

    with col1:
        label = (
            "**Bulan yang diprediksi**" if mode in ("forecast", "forecast_and_evaluate")
            else "**Bulan terakhir backtest**"
        )
        st.markdown(label)
        mc1, mc2 = st.columns(2)
        with mc1:
            target_year = st.selectbox(
                "Tahun",
                options=list(range(2020, today.year + 2)),
                index=list(range(2020, today.year + 2)).index(today.year),
            )
        with mc2:
            target_month_num = st.selectbox(
                "Bulan",
                options=list(range(1, 13)),
                format_func=lambda m: _BULAN_ID[m - 1],
                index=today.month - 1,
            )
        train_end = None
        target_month_date = date(target_year, target_month_num, 1)

        if mode in ("forecast", "forecast_and_evaluate"):
            prev_bulan = _BULAN_ID[target_month_num - 2] if target_month_num > 1 else "Desember"
            prev_tahun = target_year if target_month_num > 1 else target_year - 1
            st.caption(
                f"Training: s/d **{prev_bulan} {prev_tahun}** · "
                f"Sinyal: Open 1 {_BULAN_ID[target_month_num - 1]} {target_year} · "
                f"Prediksi: Low {_BULAN_ID[target_month_num - 1]} {target_year}"
            )
        else:
            st.caption(
                f"Backtest mencakup N bulan terakhir s/d "
                f"**{_BULAN_ID[target_month_num - 1]} {target_year}**. "
                "Tiap iterasi menggunakan Open hari pertama bulan target sebagai sinyal."
            )

        log_transform = st.checkbox(
            "Log-transform (log Y & X)",
            value=True,
            help="Stabilkan variance; output di-inverse ke Rupiah dengan bias correction.",
        )
        rolling_window_years = st.number_input(
            "Rolling window (tahun, 0 = semua data)",
            min_value=0, max_value=10, value=0,
        )

    with col2:
        max_lag_endog = st.number_input(
            "Max lag endogen (Low)", min_value=1, max_value=12, value=4,
        )
        max_lag_exog = st.number_input(
            "Max lag eksogen", min_value=1, max_value=12, value=4,
        )
        ic = st.selectbox("Information Criterion", ["aic", "bic"], index=0, disabled=True)

    # Field kondisional per mode
    eval_end = None
    backtest_months = None
    if mode == "forecast_and_evaluate":
        eval_end = st.date_input(
            "Tanggal akhir data evaluasi (fetch aktual)",
            value=today,
            max_value=today,
        )
    elif mode == "backtest":
        backtest_months = st.number_input(
            "Jumlah bulan backtest", min_value=3, max_value=36, value=12,
        )

    submitted = st.form_submit_button(
        "Mulai Prediksi", type="primary", use_container_width=True
    )

if not submitted:
    st.stop()

import calendar

if mode == "backtest":
    # Backtest: kirim tanggal 1 bulan terakhir backtest — pipeline fetch s/d situ
    train_end_str = target_month_date.strftime("%Y-%m-%d")
else:
    # Forecast / Forecast+Evaluasi: training s/d akhir bulan M-1
    prev_month = target_month_date.month - 1 or 12
    prev_year = target_month_date.year if target_month_date.month > 1 else target_month_date.year - 1
    last_day = calendar.monthrange(prev_year, prev_month)[1]
    train_end_str = date(prev_year, prev_month, last_day).strftime("%Y-%m-%d")
eval_end_str = eval_end.strftime("%Y-%m-%d") if eval_end else None

status_container = st.status("Menjalankan pipeline...", expanded=True)


def progress_cb(step, detail):
    status_container.write(f"**{step}**: {detail}")


def _df_or_info(df, empty_msg="Tidak tersedia."):
    """Tampilkan DataFrame bila ada isi, selain itu pesan info."""
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(empty_msg)


def _render_pipeline_detail(result, run_config):
    """Seksi expander bertahap Tahap 1-9 untuk laporan skripsi (mode forecast)."""
    st.divider()
    st.header("Rincian Detail Pipeline (untuk laporan)")
    st.caption(
        "Tiap tahap memuat informasi yang cukup untuk menjelaskan **mengapa** "
        "hasilnya seperti itu — cocok dibahas satu per satu di laporan."
    )

    ardl = result.get("ardl_info", {})
    adf_results = result.get("adf_results") or {}

    # Tahap 1
    daily = result.get("daily")
    with st.expander("🔽 Tahap 1 — Pengambilan & Kualitas Data Harian"):
        st.markdown(explain.NARASI[1])
        if daily is not None and not daily.empty:
            st.write(
                f"Rentang: **{daily.index.min().date()}** s/d "
                f"**{daily.index.max().date()}** · Jumlah hari: **{len(daily)}**"
            )
            st.dataframe(daily.tail(10).reset_index(), use_container_width=True)
            st.markdown("**Ringkasan statistik OHLCV**")
            st.dataframe(daily.describe(), use_container_width=True)
        else:
            st.info("Data harian tidak tersedia.")

    # Tahap 2
    monthly_full = result.get("monthly_full")
    monthly = result.get("monthly_history")
    with st.expander("🔽 Tahap 2 — Resampling ke Bulanan & Rolling Window"):
        st.markdown(explain.NARASI[2])
        n_full = len(monthly_full) if monthly_full is not None else 0
        n_used = len(monthly) if monthly is not None else 0
        st.write(
            f"Baris bulanan tersedia: **{n_full}** · "
            f"dipakai untuk training: **{n_used}**"
        )
        if monthly is not None and not monthly.empty:
            st.dataframe(
                monthly.tail(12).reset_index(), use_container_width=True
            )

    # Tahap 3
    with st.expander("🔽 Tahap 3 — Transformasi Variabel (log)"):
        st.markdown(explain.NARASI[3])
        st.write(
            f"Log-transform: **{'Aktif' if run_config.get('log_transform') else 'Tidak'}**"
        )

    # Tahap 4
    with st.expander("🔽 Tahap 4 — Uji Stasioneritas (ADF)"):
        st.markdown(explain.NARASI[4])
        _df_or_info(explain.adf_to_df(adf_results), "Hasil ADF tidak tersedia.")
        if adf_results.get("_any_i2"):
            st.warning(
                "⚠️ Terdeteksi variabel I(2) — ARDL bounds test menjadi tidak valid."
            )

    # Tahap 5
    with st.expander("🔽 Tahap 5 — Pemilihan Lag (Endogen & Eksogen) & Estimasi ARDL"):
        st.markdown(explain.NARASI[5])
        if ardl.get("endog_lag") is not None:
            st.write(
                f"Lag endogen terpilih (p): **{ardl['endog_lag']}** · "
                f"Order eksogen: **{ardl.get('exog_orders')}**"
            )
        if ardl.get("train_r2") is not None:
            st.write(f"R² training: **{ardl['train_r2']:.4f}**")

        lag_ic = result.get("lag_ic_table")
        st.markdown(
            "**Tabel kandidat lag (Information Criterion)** — baris ✓ = IC terkecil → "
            "kombinasi p & q terpilih."
        )
        if lag_ic is not None and not lag_ic.empty:
            st.caption(
                f"Total {len(lag_ic)} kombinasi dievaluasi "
                f"(grid: max lag endog={run_config.get('max_lag_endog')}, "
                f"max lag exog={run_config.get('max_lag_exog')})."
            )
            st.dataframe(lag_ic, use_container_width=True, hide_index=True)
        else:
            st.info("Tabel IC tidak tersedia.")

        st.markdown("**Koefisien model ARDL**")
        _df_or_info(result.get("coef_table"), "Koefisien tidak tersedia.")

    # Tahap 6
    with st.expander("🔽 Tahap 6 — Uji Kointegrasi (Bounds Test Pesaran-Shin-Smith)"):
        st.markdown(explain.NARASI[6])
        if ardl.get("bounds_f_stat") is not None:
            st.write(
                f"F-statistik: **{ardl['bounds_f_stat']:.4f}** · "
                f"Verdict: **{ardl.get('cointegration')}**"
            )
        st.markdown("**Batas kritis (pita keputusan)**")
        _df_or_info(
            explain.bounds_table_to_df(ardl), "Batas kritis tidak tersedia."
        )

    # Tahap 7
    ecm = result.get("ecm_info")
    with st.expander("🔽 Tahap 7 — Diagnostik ECM (Error Correction)"):
        st.markdown(explain.NARASI[7])
        if ecm:
            c1, c2, c3 = st.columns(3)
            c1.metric("λ (speed of adjustment)", f"{ecm['lambda']:.6f}")
            hl = ecm.get("half_life_months")
            c2.metric("Half-life", f"{hl:.2f} bln" if hl else "-")
            c3.metric("p-value λ", f"{ecm['p']:.4f}")
            st.markdown("**Koefisien jangka panjang**")
            _df_or_info(explain.longrun_to_df(ecm))
            st.markdown("**Diagnostik residual**")
            _df_or_info(explain.residual_diag_to_df(ecm))
        else:
            st.info("Tidak kointegrasi → ECM tidak diinterpretasikan.")

    # Tahap 8
    var_info = result.get("var_info") or {}
    exog_future = result.get("exog_future")
    with st.expander("🔽 Tahap 8 — Proyeksi Variabel Eksogen (VAR)"):
        st.markdown(explain.NARASI[8])
        if var_info.get("k_ar") is not None:
            txt = f"Lag VAR terpilih (k_ar): **{var_info['k_ar']}**"
            if var_info.get("fallback_p1"):
                txt += " (fallback p=1)"
            st.write(txt)
        var_only = var_info.get("exog_var_only")
        current_open = result.get("current_open")
        if var_only is not None and not var_only.empty:
            st.markdown("**Proyeksi VAR murni (sebelum injeksi Open)**")
            st.dataframe(
                var_only.reset_index(), use_container_width=True, hide_index=True
            )
        if exog_future is not None and not exog_future.empty:
            st.markdown("**Eksogen final (Open aktual diinjeksikan)**")
            st.dataframe(
                exog_future.reset_index(), use_container_width=True, hide_index=True
            )
        if current_open is not None:
            st.caption(f"Open aktual hari pertama bulan target: Rp {current_open:,.0f}")

    # Tahap 9
    fc = result.get("forecast")
    fd = result.get("forecast_detail") or {}
    with st.expander("🔽 Tahap 9 — Forecasting Akhir + Interval Keyakinan"):
        st.markdown(explain.NARASI[9])
        if fd:
            st.write(
                f"ŷ (skala log): **{fd.get('yhat_log')}** · "
                f"σ² (bias correction): **{fd.get('sigma2')}**"
            )
        if fc is not None and not fc.empty:
            row = fc.iloc[0]
            st.write(
                f"Inverse-log → **Prediksi Low: Rp {row['Predicted_Low']:,.0f}** · "
                f"CI 95%: [Rp {row['CI_Lower']:,.0f} , Rp {row['CI_Upper']:,.0f}] · "
                f"Lebar CI: Rp {(row['CI_Upper'] - row['CI_Lower']):,.0f}"
            )
        ev = result.get("eval_result")
        if ev:
            st.write(
                f"Aktual: Rp {ev['actual_low']:,.0f} · Error: {ev['error_pct']:.2f}% · "
                f"Dalam CI 95%: {'Ya' if ev['in_ci'] else 'Tidak'}"
            )

    _render_download_buttons(result, run_config)


def _render_backtest_detail(result, run_config):
    """Expander tunggal untuk mode backtest."""
    st.divider()
    st.header("Rincian Detail Pipeline (untuk laporan)")
    with st.expander("🔽 Rincian Pipeline (Backtest Walk-Forward)", expanded=False):
        st.markdown(
            "Setiap iterasi backtest mengulang **seluruh tahap** (data → "
            "stasioneritas → pemilihan lag → estimasi → kointegrasi → proyeksi VAR "
            "→ forecast) memakai data s/d bulan target, lalu membandingkan prediksi "
            "dengan Low aktual. Kolom diagnostik per-bulan tersedia di tabel hasil "
            "(endog_lag, exog_orders, bounds_F, cointegration)."
        )
    _render_download_buttons(result, run_config)


def _render_download_buttons(result, run_config):
    """Tombol download laporan pipeline (Markdown + PDF)."""
    st.subheader("Unduh Laporan Pipeline")
    try:
        md = explain.build_pipeline_report_md(result, run_config)
        st.download_button(
            "⬇️ Download .md",
            md.encode("utf-8"),
            file_name="laporan_pipeline.md",
            mime="text/markdown",
        )
    except Exception as e:
        st.warning(f"Gagal membuat Markdown: {e}")
    try:
        pdf = explain.build_pipeline_report_pdf(result, run_config)
        st.download_button(
            "⬇️ Download .pdf",
            pdf,
            file_name="laporan_pipeline.pdf",
            mime="application/pdf",
        )
    except Exception as e:
        st.warning(f"Gagal membuat PDF: {e}")


common_kwargs = dict(
    log_transform=log_transform,
    rolling_window_years=int(rolling_window_years),
    max_lag_endog=int(max_lag_endog),
    max_lag_exog=int(max_lag_exog),
    ic=ic,
    var_maxlag=int(max_lag_exog),
    progress_callback=progress_cb,
)

try:
    if mode == "forecast":
        target_month_str = target_month_date.strftime("%Y-%m")
        result = run_monthly_forecast(
            train_end_str, target_month=target_month_str, **common_kwargs
        )
    elif mode == "forecast_and_evaluate":
        target_month_str = target_month_date.strftime("%Y-%m")
        result = run_monthly_forecast_and_evaluate(
            train_end_str, eval_end_str,
            target_month=target_month_str, **common_kwargs
        )
    else:  # backtest
        result = run_monthly_backtest(
            train_end_str, backtest_months=int(backtest_months), **common_kwargs
        )

    status_container.write("**save**: Menyimpan run + config ke database...")

    session = get_session()
    run_config = {
        "mode": mode,
        "train_end_date": date.fromisoformat(train_end_str),
        "eval_end_date": eval_end,
        "log_transform": log_transform,
        "rolling_window_years": int(rolling_window_years),
        "max_lag_endog": int(max_lag_endog),
        "max_lag_exog": int(max_lag_exog),
        "ic": ic,
        "trend": TREND,
        "var_maxlag": int(max_lag_exog),
        "backtest_months": int(backtest_months) if backtest_months else None,
    }
    run_id = save_model_run(session, run_config)

    # Simpan data harga harian ke price_history (untuk history di Dashboard).
    daily = result.get("daily")
    if daily is not None:
        daily_to_save = daily.reset_index().rename(columns={"index": "Time"})
        if "Time" not in daily_to_save.columns:
            daily_to_save = daily_to_save.rename(
                columns={daily_to_save.columns[0]: "Time"}
            )
        save_price_history(session, daily_to_save)

    if mode == "backtest":
        metrics = result["metrics"]
        save_backtest_predictions(session, run_id, result["results_df"])
        save_backtest_evaluations(session, run_id, result["results_df"])
        # Simpan lag dari iterasi terakhir yang sukses sebagai representasi model
        valid_rows = result["results_df"].dropna(subset=["endog_lag"])
        last_ardl_info = None
        if not valid_rows.empty:
            last = valid_rows.iloc[-1]
            last_ardl_info = {
                "endog_lag":    int(last["endog_lag"]),
                "exog_orders":  str(last["exog_orders"]),
                "bounds_f_stat": float(last["bounds_F"]) if "bounds_F" in last and pd.notnull(last["bounds_F"]) else None,
                "cointegration": last["cointegration"],
                "lambda_ecm":   None,
                "half_life_months": None,
                "train_r2":     None,
            }
        update_model_run_result(
            session, run_id, ardl_info=last_ardl_info, metrics=metrics, status="success"
        )
    else:
        save_monthly_predictions(session, run_id, result["forecast"])
        update_model_run_result(
            session, run_id, ardl_info=result["ardl_info"], status="success"
        )
        eval_result = result.get("eval_result")
        if eval_result is not None:
            save_single_evaluation(session, run_id, eval_result)
            err = eval_result
            metrics = {
                "mape": err["error_pct"],
                "ci_coverage": 100.0 if err["in_ci"] else 0.0,
            }
            update_model_run_result(session, run_id, metrics=metrics)

    session.close()
    status_container.update(label="Pipeline selesai!", state="complete")
    st.success(f"Prediksi berhasil! (Run ID: {run_id})")

    # ====================== TAMPILKAN HASIL ======================
    if mode == "backtest":
        metrics = result["metrics"]
        if metrics is None:
            st.error("Semua iterasi backtest gagal — tidak ada metrik.")
        else:
            st.subheader("Metrik Backtest (walk-forward)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAPE", f"{metrics['mape']:.2f}%")
            m2.metric("RMSE", f"Rp {metrics['rmse']:,.0f}")
            m3.metric("MAE", f"Rp {metrics['mae']:,.0f}")
            m4.metric("CI Coverage", f"{metrics['ci_coverage']:.1f}%")
            st.caption(
                f"Sample: {metrics['n']} bulan | "
                f"Cointegrated: {metrics['coint_rate']:.0f}% iterasi | "
                f"R²: {metrics['r2']:.4f}"
            )

        if metrics is not None:
            st.subheader("Chart Backtest")
            st.altair_chart(
                backtest_chart(result["results_df"]),
                use_container_width=True,
            )
            st.caption(
                "Garis hijau = aktual · garis merah putus-putus = prediksi · "
                "pita = 95% CI. Arahkan kursor untuk detail; scroll untuk zoom."
            )

        st.subheader("Tabel Hasil Backtest")
        bt = result["results_df"].copy()
        bt["month"] = pd.to_datetime(bt["month"]).dt.strftime("%Y-%m")
        for col in ["actual_low", "predicted_low", "ci_lower", "ci_upper"]:
            bt[col] = bt[col].apply(
                lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A"
            )
        bt["error_pct"] = bt["error_pct"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A"
        )
        bt["bounds_F"] = bt["bounds_F"].apply(
            lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
        )
        st.dataframe(bt, use_container_width=True, hide_index=True)

        _render_backtest_detail(result, run_config)
    else:
        ardl = result["ardl_info"]
        st.subheader("Info Model")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write(f"**Model**: ARDL(p={ardl['endog_lag']})")
            st.write(f"**Exog orders**: {ardl['exog_orders']}")
        with c2:
            st.write(f"**Bounds F-stat**: {ardl['bounds_f_stat']:.4f}")
            st.write(f"**Kointegrasi**: {ardl['cointegration']}")
        with c3:
            lam = ardl.get("lambda_ecm")
            st.write(f"**ECM λ**: {lam:.6f}" if lam is not None else "**ECM λ**: -")
            hl = ardl.get("half_life_months")
            st.write(
                f"**Half-life**: {hl:.2f} bulan" if hl else "**Half-life**: -"
            )
        st.caption(f"R² training: {ardl['train_r2']:.4f}")

        fc = result["forecast"]
        target_month = fc.index[0].strftime("%Y-%m")
        pred_row = fc.iloc[0]
        eval_result = result.get("eval_result")

        st.subheader(f"Prediksi Bulanan ({target_month})")
        monthly_history = result["monthly_history"]
        delta_txt = None
        if len(monthly_history) >= 3:
            avg_recent = monthly_history["Low"].tail(3).mean()
            diff_pct = (pred_row["Predicted_Low"] - avg_recent) / avg_recent * 100
            delta_txt = f"{diff_pct:+.1f}% vs rata-rata 3 bln"
        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Prediksi Low", f"Rp {pred_row['Predicted_Low']/1e9:,.3f} M",
            delta=delta_txt, delta_color="inverse",
        )
        p2.metric("CI Bawah (95%)", f"Rp {pred_row['CI_Lower']/1e9:,.3f} M")
        p3.metric("CI Atas (95%)", f"Rp {pred_row['CI_Upper']/1e9:,.3f} M")

        st.subheader("Chart Forecast")
        eval_chart_df = None
        if eval_result is not None:
            eval_chart_df = pd.DataFrame(
                [
                    {
                        "Target_Month": eval_result["target_month"],
                        "Actual_Low": eval_result["actual_low"],
                    }
                ]
            )
        st.altair_chart(
            forecast_with_history_chart(monthly_history, fc, eval_chart_df),
            use_container_width=True,
        )
        st.caption(
            "Garis biru = low historis bulanan · berlian merah = prediksi · "
            "lingkaran hijau = aktual (bila tersedia)."
        )

        if eval_result is not None:
            st.subheader("Evaluasi vs Aktual")
            e1, e2, e3 = st.columns(3)
            e1.metric("Actual Low", f"Rp {eval_result['actual_low']/1e9:,.3f} M")
            e2.metric("Error", f"{eval_result['error_pct']:.2f}%")
            e3.metric("Dalam CI 95%", "✅ Ya" if eval_result["in_ci"] else "❌ Tidak")
        elif mode == "forecast_and_evaluate":
            st.info("Data aktual untuk bulan target belum tersedia.")

        _render_pipeline_detail(result, run_config)

except Exception as e:
    status_container.update(label="Pipeline gagal!", state="error")
    st.error(f"Error: {e}")
    raise
