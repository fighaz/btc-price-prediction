"""
Dashboard - Ringkasan prediksi terbaru
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from db.engine import get_session
from db.repository import get_latest_prediction, get_evaluation_for_run

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("Dashboard")

session = get_session()
run_info, pred_df = get_latest_prediction(session)

if run_info is None or pred_df.empty:
    st.info("Belum ada prediksi. Buka halaman **Prediksi Baru** untuk menjalankan prediksi pertama.")
    session.close()
    st.stop()

# Info model
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", f"ARDL(p={run_info['p_lags']}, q={run_info['q_lags']})")
with col2:
    st.metric("R² Training", f"{run_info['train_r2']:.4f}" if run_info['train_r2'] else "-")
with col3:
    st.metric("Alpha", f"{run_info['alpha']}")
with col4:
    st.metric("Forecast Days", f"{run_info['forecast_days']}")

if run_info.get("eval_mape"):
    col5, col6 = st.columns(2)
    with col5:
        st.metric("MAPE (Evaluasi)", f"{run_info['eval_mape']:.2f}%")
    with col6:
        st.metric("CI Coverage", f"{run_info['eval_ci_coverage']:.1f}%" if run_info.get('eval_ci_coverage') else "-")

st.divider()

# Chart prediksi
st.subheader("Prediksi Terbaru")

pred_df["Tanggal"] = pd.to_datetime(pred_df["Tanggal"])

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(pred_df["Tanggal"], pred_df["Predicted_Low"] / 1e9,
        label="Prediksi Low", color="#E91E63", linewidth=2, marker="o", markersize=3)
ax.fill_between(pred_df["Tanggal"],
                pred_df["Confidence_Lower"] / 1e9,
                pred_df["Confidence_Upper"] / 1e9,
                alpha=0.2, color="#E91E63", label="95% CI")

# Overlay evaluasi jika ada
eval_df = get_evaluation_for_run(session, run_info["id"])
if not eval_df.empty:
    eval_df["Tanggal"] = pd.to_datetime(eval_df["Tanggal"])
    ax.plot(eval_df["Tanggal"], eval_df["Actual_Low"] / 1e9,
            label="Aktual", color="#4CAF50", linewidth=2, marker="s", markersize=3)

ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga Low (Miliar IDR)")
ax.set_title("Forecast Harga Terendah BTC/IDR")
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Tabel prediksi
st.subheader("Tabel Prediksi")
display_df = pred_df.copy()
display_df["Predicted_Low"] = display_df["Predicted_Low"].apply(lambda x: f"Rp {x:,.0f}")
display_df["Confidence_Lower"] = display_df["Confidence_Lower"].apply(lambda x: f"Rp {x:,.0f}")
display_df["Confidence_Upper"] = display_df["Confidence_Upper"].apply(lambda x: f"Rp {x:,.0f}")
display_df["Tanggal"] = display_df["Tanggal"].dt.strftime("%Y-%m-%d")
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.caption(f"Run ID: {run_info['id']} | Tanggal run: {run_info['run_at']}")
session.close()
