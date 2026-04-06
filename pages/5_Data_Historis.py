"""
Data Historis - Eksplorasi data harga mentah BTC/IDR
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

from db.engine import init_db, get_session
from db.repository import save_price_history, get_price_history, get_latest_price_date
from src.data import fetch_btc_data
from src.statistics import run_adf_test

init_db()

st.set_page_config(page_title="Data Historis", page_icon="📁", layout="wide")
st.title("Data Historis BTC/IDR")

session = get_session()

# Info data di DB
latest_date = get_latest_price_date(session)
if latest_date:
    st.info(f"Data terakhir di database: **{latest_date}**")
else:
    st.warning("Database kosong. Klik tombol di bawah untuk mengambil data.")

# Tombol update data
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Perbarui Data", type="primary"):
        with st.spinner("Mengambil data dari Indodax API..."):
            df = fetch_btc_data(to_date=date.today().strftime("%Y-%m-%d"))
            if df is not None:
                count = save_price_history(session, df)
                st.success(f"Berhasil! {count} baris baru ditambahkan.")
                st.rerun()
            else:
                st.error("Gagal mengambil data dari API.")

# Load data dari DB
price_df = get_price_history(session)

if price_df.empty:
    st.info("Belum ada data. Klik **Perbarui Data** untuk mengambil data dari API.")
    session.close()
    st.stop()

# Filter tanggal
st.subheader("Filter Data")
fcol1, fcol2 = st.columns(2)
with fcol1:
    start_date = st.date_input("Dari", value=price_df["Time"].min().date())
with fcol2:
    end_date = st.date_input("Sampai", value=price_df["Time"].max().date())

filtered = price_df[
    (price_df["Time"] >= pd.Timestamp(start_date)) &
    (price_df["Time"] <= pd.Timestamp(end_date))
]

# Statistik
st.subheader("Statistik Data")
scol1, scol2, scol3, scol4 = st.columns(4)
with scol1:
    st.metric("Total Records", f"{len(filtered):,}")
with scol2:
    st.metric("Rentang", f"{start_date} - {end_date}")
with scol3:
    st.metric("Low Min", f"Rp {filtered['Low'].min():,.0f}" if not filtered.empty else "-")
with scol4:
    st.metric("Low Max", f"Rp {filtered['Low'].max():,.0f}" if not filtered.empty else "-")

# Chart harga
st.subheader("Chart Harga")
chart_col = st.selectbox("Kolom", ["Low", "Close", "Open", "High"], index=0)

if not filtered.empty:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(filtered["Time"], filtered[chart_col] / 1e9, color="#2196F3", linewidth=1)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel(f"{chart_col} (Miliar IDR)")
    ax.set_title(f"BTC/IDR {chart_col} Price")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tabel data
st.subheader("Tabel Data")
display = filtered.copy()
display["Time"] = display["Time"].dt.strftime("%Y-%m-%d")
for col in ["Open", "High", "Low", "Close"]:
    display[col] = display[col].apply(lambda x: f"Rp {x:,.0f}")
display["Volume"] = display["Volume"].apply(lambda x: f"{x:,.0f}")
st.dataframe(display, use_container_width=True, hide_index=True, height=400)

# ADF test on demand
st.divider()
st.subheader("ADF Stationarity Test")
if st.button("Jalankan ADF Test"):
    if not filtered.empty:
        with st.spinner("Menjalankan ADF test..."):
            results = run_adf_test(filtered)

            adf_data = []
            for col, r in results.items():
                row = {
                    "Variabel": col,
                    "ADF Statistic": f"{r['adf_statistic']:.4f}",
                    "p-value": f"{r['p_value']:.4f}",
                    "Status": r["status"],
                }
                if "diff_status" in r:
                    row["First Diff"] = r["diff_status"]
                adf_data.append(row)

            st.dataframe(pd.DataFrame(adf_data), use_container_width=True, hide_index=True)
            st.caption("ARDL valid untuk campuran variabel I(0) dan I(1)")
    else:
        st.warning("Tidak ada data untuk ditest.")

session.close()
