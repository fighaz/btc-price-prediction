"""
Prediksi Baru - Jalankan pipeline prediksi ARDL
"""
import sys
import logging
import streamlit as st
from datetime import date, timedelta

from db.engine import init_db, get_session
from db.repository import (
    save_model_run, save_predictions, save_evaluation,
    update_model_run_status, save_price_history,
)
from src.pipeline import run_forecast, run_forecast_and_evaluate
from config import DEFAULT_MIN_LAG

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Prediksi Baru", page_icon="🔮", layout="wide")
st.title("Prediksi Baru")
st.markdown("Jalankan pipeline prediksi ARDL dengan parameter yang bisa disesuaikan.")

init_db()

# Form input
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        train_end = st.date_input(
            "Tanggal akhir data training",
            value=date.today() - timedelta(days=1),
            max_value=date.today(),
        )
        forecast_days = st.number_input(
            "Jumlah hari prediksi",
            min_value=1, max_value=90, value=31,
        )

    with col2:
        do_eval = st.checkbox("Jalankan evaluasi juga?", value=False)
        eval_end = st.date_input(
            "Tanggal akhir data evaluasi",
            value=date.today(),
            disabled=not do_eval,
        )
        min_lag = st.number_input(
            "Minimum lag",
            min_value=1, max_value=5, value=DEFAULT_MIN_LAG,
        )

    submitted = st.form_submit_button("Mulai Prediksi", type="primary", use_container_width=True)

if submitted:
    train_end_str = train_end.strftime("%Y-%m-%d")
    eval_end_str = eval_end.strftime("%Y-%m-%d") if do_eval else None

    status_container = st.status("Menjalankan pipeline prediksi...", expanded=True)

    def progress_cb(step, detail):
        status_container.write(f"**{step}**: {detail}")

    try:
        if do_eval:
            result = run_forecast_and_evaluate(
                train_end_str, forecast_days,
                eval_end_date=eval_end_str,
                min_lag=min_lag,
                progress_callback=progress_cb,
            )
        else:
            result = run_forecast(
                train_end_str, forecast_days,
                min_lag=min_lag,
                progress_callback=progress_cb,
            )

        status_container.write("**save**: Menyimpan ke database...")

        # Save ke database
        session = get_session()

        # Save price history
        save_price_history(session, result["df"])

        # Save model run
        train_info = result["train_info"]
        run_id = save_model_run(session, {
            "mode": "forecast_and_evaluate" if do_eval else "forecast_only",
            "train_end_date": train_end,
            "forecast_days": forecast_days,
            "p_lags": result["p_lags"],
            "q_lags": result["q_lags"],
            "min_lag": min_lag,
            "alpha": train_info["best_alpha"],
            "n_features": train_info["n_features"],
            "train_r2": train_info["train_r2"],
            "residual_std": train_info["residual_std"],
        })

        # Save predictions
        save_predictions(session, run_id, result["forecast_df"])

        # Save evaluation jika ada
        metrics = result.get("metrics")
        merged = result.get("merged_eval")
        if metrics and merged is not None:
            save_evaluation(session, run_id, merged)
            update_model_run_status(session, run_id, "success", metrics)
        else:
            update_model_run_status(session, run_id, "success")

        session.close()
        status_container.update(label="Pipeline selesai!", state="complete")

        # Tampilkan hasil
        st.success(f"Prediksi berhasil! (Run ID: {run_id})")

        # Info model
        st.subheader("Info Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Model**: ARDL(p={result['p_lags']}, q={result['q_lags']})")
            st.write(f"**Alpha**: {train_info['best_alpha']}")
        with col2:
            st.write(f"**R² Training**: {train_info['train_r2']:.4f}")
            st.write(f"**Residual Std**: Rp {train_info['residual_std']:,.0f}")
        with col3:
            st.write(f"**Features**: {train_info['n_features']}")
            st.write(f"**Data**: {train_info['date_start']} - {train_info['date_end']}")

        # Chart
        if result.get("figure"):
            st.subheader("Chart Forecast")
            st.pyplot(result["figure"])

        # Metrics evaluasi
        if metrics:
            st.subheader("Metrik Evaluasi")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            with mc2:
                st.metric("RMSE", f"Rp {metrics['rmse']:,.0f}")
            with mc3:
                st.metric("MAE", f"Rp {metrics['mae']:,.0f}")
            with mc4:
                st.metric("CI Coverage", f"{metrics['ci_coverage']:.1f}%")

        # Tabel prediksi
        st.subheader("Tabel Prediksi")
        display = result["forecast_df"].copy()
        display["Time"] = display["Time"].dt.strftime("%Y-%m-%d")
        display["Predicted_Low"] = display["Predicted_Low"].apply(lambda x: f"Rp {x:,.0f}")
        display["Confidence_Lower"] = display["Confidence_Lower"].apply(lambda x: f"Rp {x:,.0f}")
        display["Confidence_Upper"] = display["Confidence_Upper"].apply(lambda x: f"Rp {x:,.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    except Exception as e:
        status_container.update(label="Pipeline gagal!", state="error")
        st.error(f"Error: {str(e)}")
        raise
