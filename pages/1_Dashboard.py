"""
Dashboard - Ringkasan prediksi terbaru + low historis bulanan (ARDL-ECM monthly)
"""
import pandas as pd
import streamlit as st

from db.engine import get_session
from db.repository import get_latest_run, get_evaluation_for_run, get_price_history
from src.ardl_ecm.data import resample_to_monthly
from src.ardl_ecm.charts import forecast_with_history_chart

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")
st.caption("Ringkasan prediksi harga terendah bulanan BTC/IDR — model ARDL-ECM")

session = get_session()
run_info, pred_df = get_latest_run(session)

if run_info is None or pred_df.empty:
    st.info(
        "Belum ada prediksi. Buka halaman **🔮 Prediksi Baru** untuk menjalankan "
        "prediksi pertama."
    )
    session.close()
    st.stop()

pred_df = pred_df.copy()
pred_df["Target_Month"] = pd.to_datetime(pred_df["Target_Month"])

# ============================================================================
# Low historis bulanan dari price_history (di-resample)
# ============================================================================
price_df = get_price_history(session)
monthly_history = pd.DataFrame()
if not price_df.empty:
    daily_indexed = price_df.set_index("Time")
    monthly_history = resample_to_monthly(daily_indexed, drop_partial=False)

# ============================================================================
# Kartu metrik utama
# ============================================================================
pred_row = pred_df.iloc[0]
target_month = pred_row["Target_Month"].strftime("%Y-%m")
pred_low = pred_row["Predicted_Low"]

# delta vs rata-rata low 3 bulan terakhir
delta_txt = None
if not monthly_history.empty and len(monthly_history) >= 3:
    avg_recent = monthly_history["Low"].tail(3).mean()
    diff_pct = (pred_low - avg_recent) / avg_recent * 100
    delta_txt = f"{diff_pct:+.1f}% vs rata-rata 3 bln"

st.subheader(f"Prediksi Bulan {target_month}")
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Prediksi Low", f"Rp {pred_low/1e9:,.3f} M", delta=delta_txt,
    delta_color="inverse",
)
m2.metric(
    "CI Bawah (95%)",
    f"Rp {pred_row['CI_Lower']/1e9:,.3f} M"
    if pd.notnull(pred_row["CI_Lower"]) else "-",
)
m3.metric(
    "CI Atas (95%)",
    f"Rp {pred_row['CI_Upper']/1e9:,.3f} M"
    if pd.notnull(pred_row["CI_Upper"]) else "-",
)
f = run_info.get("bounds_f_stat")
m4.metric(
    "Kointegrasi",
    run_info.get("cointegration") or "-",
    delta=f"F={f:.2f}" if f is not None else None,
    delta_color="off",
)

# Ringkasan naratif
if pd.notnull(pred_row["CI_Lower"]) and pd.notnull(pred_row["CI_Upper"]):
    st.markdown(
        f"> Model memprediksi harga **terendah** BTC/IDR pada **{target_month}** "
        f"sekitar **Rp {pred_low:,.0f}**, dengan rentang kemungkinan "
        f"(95% CI) **Rp {pred_row['CI_Lower']:,.0f}** – "
        f"**Rp {pred_row['CI_Upper']:,.0f}**."
    )

# Metrik model sekunder
with st.expander("Detail model"):
    d1, d2, d3, d4 = st.columns(4)
    p = run_info.get("endog_lag")
    d1.metric("Model", f"ARDL(p={p})" if p is not None else "ARDL-ECM")
    d2.metric("Mode run", run_info["mode"])
    lam = run_info.get("lambda_ecm")
    d3.metric("ECM λ", f"{lam:.4f}" if lam is not None else "-")
    r2 = run_info.get("train_r2")
    d4.metric("R² training", f"{r2:.4f}" if r2 is not None else "-")
    if run_info.get("eval_mape") is not None:
        e1, e2 = st.columns(2)
        e1.metric("MAPE (Evaluasi)", f"{run_info['eval_mape']:.2f}%")
        ci = run_info.get("eval_ci_coverage")
        e2.metric("CI Coverage", f"{ci:.1f}%" if ci is not None else "-")

st.divider()

# ============================================================================
# Chart utama: history low + prediksi
# ============================================================================
st.subheader("Tren Harga Terendah Bulanan")

eval_df = get_evaluation_for_run(session, run_info["id"])

if monthly_history.empty:
    st.warning(
        "Data historis belum tersedia di database. Jalankan **🔮 Prediksi Baru** "
        "untuk mengisi data harga, atau gunakan tombol *Perbarui Data* di "
        "**📁 Data Historis**."
    )
else:
    chart = forecast_with_history_chart(
        monthly_history, pred_df, eval_df if not eval_df.empty else None
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Garis biru = low historis bulanan · berlian merah = prediksi · "
        "lingkaran hijau = aktual (bila tersedia). Arahkan kursor untuk detail; "
        "scroll untuk zoom."
    )

# ============================================================================
# Tabel prediksi
# ============================================================================
st.subheader("Tabel Prediksi")
display_df = pred_df.copy()
display_df["Target_Month"] = display_df["Target_Month"].dt.strftime("%Y-%m")
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Target_Month": st.column_config.TextColumn("Bulan Target"),
        "Predicted_Low": st.column_config.NumberColumn(
            "Prediksi Low", format="Rp %.0f"
        ),
        "CI_Lower": st.column_config.NumberColumn("CI Bawah", format="Rp %.0f"),
        "CI_Upper": st.column_config.NumberColumn("CI Atas", format="Rp %.0f"),
    },
)

st.caption(f"Run ID: {run_info['id']} | Tanggal run: {run_info['run_at']}")
session.close()
