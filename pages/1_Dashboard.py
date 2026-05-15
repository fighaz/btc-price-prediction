"""
Dashboard - Ringkasan prediksi terbaru (ARDL-ECM monthly)
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from db.engine import get_session
from db.repository import get_latest_run, get_evaluation_for_run

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Dashboard")

session = get_session()
run_info, pred_df = get_latest_run(session)

if run_info is None or pred_df.empty:
    st.info(
        "Belum ada prediksi. Buka halaman **Prediksi Baru** untuk menjalankan "
        "prediksi pertama."
    )
    session.close()
    st.stop()

# Info model
col1, col2, col3, col4 = st.columns(4)
with col1:
    p = run_info.get("endog_lag")
    st.metric("Model", f"ARDL(p={p})" if p is not None else "ARDL-ECM")
with col2:
    st.metric("Mode", run_info["mode"])
with col3:
    f = run_info.get("bounds_f_stat")
    st.metric("Bounds F-stat", f"{f:.4f}" if f is not None else "-")
with col4:
    st.metric("Kointegrasi", run_info.get("cointegration") or "-")

if run_info.get("eval_mape") is not None:
    c5, c6, c7 = st.columns(3)
    with c5:
        st.metric("MAPE (Evaluasi)", f"{run_info['eval_mape']:.2f}%")
    with c6:
        ci = run_info.get("eval_ci_coverage")
        st.metric("CI Coverage", f"{ci:.1f}%" if ci is not None else "-")
    with c7:
        lam = run_info.get("lambda_ecm")
        st.metric("ECM λ", f"{lam:.4f}" if lam is not None else "-")

st.divider()
st.subheader("Prediksi Terbaru")

pred_df = pred_df.copy()
pred_df["Target_Month"] = pd.to_datetime(pred_df["Target_Month"])
months_str = pred_df["Target_Month"].dt.strftime("%Y-%m")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    months_str, pred_df["Predicted_Low"] / 1e9,
    label="Prediksi Low", color="#E91E63", linewidth=2, marker="o", markersize=6,
)
if pred_df["CI_Lower"].notna().all() and pred_df["CI_Upper"].notna().all():
    ax.fill_between(
        months_str, pred_df["CI_Lower"] / 1e9, pred_df["CI_Upper"] / 1e9,
        alpha=0.2, color="#E91E63", label="95% CI",
    )

# Overlay evaluasi jika ada
eval_df = get_evaluation_for_run(session, run_info["id"])
if not eval_df.empty:
    eval_df = eval_df.copy()
    eval_df["Target_Month"] = pd.to_datetime(eval_df["Target_Month"])
    ax.plot(
        eval_df["Target_Month"].dt.strftime("%Y-%m"),
        eval_df["Actual_Low"] / 1e9,
        label="Aktual", color="#4CAF50", linewidth=2, marker="s", markersize=6,
    )

ax.set_xlabel("Bulan")
ax.set_ylabel("Harga Low (Miliar IDR)")
ax.set_title("Forecast Harga Terendah Bulanan BTC/IDR")
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.subheader("Tabel Prediksi")
display_df = pred_df.copy()
display_df["Target_Month"] = display_df["Target_Month"].dt.strftime("%Y-%m")
for col in ["Predicted_Low", "CI_Lower", "CI_Upper"]:
    display_df[col] = display_df[col].apply(
        lambda x: f"Rp {x:,.0f}" if pd.notnull(x) else "N/A"
    )
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(f"Run ID: {run_info['id']} | Tanggal run: {run_info['run_at']}")
session.close()
