"""
Data Historis - Eksplorasi data harga BTC/IDR (harian & bulanan)
"""
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from db.engine import init_db, get_session
from db.repository import save_price_history, get_price_history, get_latest_price_date
from src.ardl_ecm import fetch_btc_daily, resample_to_monthly
from src.ardl_ecm.model import prepare_variables, check_stationarity

init_db()

st.set_page_config(page_title="Data Historis", page_icon="📁", layout="wide")
st.title("Data Historis BTC/IDR")

session = get_session()

latest_date = get_latest_price_date(session)
if latest_date:
    st.info(f"Data terakhir di database: **{latest_date}**")
else:
    st.warning("Database kosong. Klik tombol di bawah untuk mengambil data.")

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Perbarui Data", type="primary"):
        with st.spinner("Mengambil data dari Indodax API..."):
            df = fetch_btc_daily(to_date=date.today().strftime("%Y-%m-%d"))
            df = df.reset_index().rename(columns={"index": "Time"})
            count = save_price_history(session, df)
            st.success(f"Berhasil! {count} baris baru ditambahkan.")
            st.rerun()

price_df = get_price_history(session)
if price_df.empty:
    st.info("Belum ada data. Klik **Perbarui Data** untuk mengambil data dari API.")
    session.close()
    st.stop()

# Toggle frekuensi
freq = st.radio("Frekuensi data", ["Harian", "Bulanan"], horizontal=True)

# Filter tanggal
st.subheader("Filter Data")
fcol1, fcol2 = st.columns(2)
with fcol1:
    start_date = st.date_input("Dari", value=price_df["Time"].min().date())
with fcol2:
    end_date = st.date_input("Sampai", value=price_df["Time"].max().date())

filtered = price_df[
    (price_df["Time"] >= pd.Timestamp(start_date))
    & (price_df["Time"] <= pd.Timestamp(end_date))
].copy()

if freq == "Bulanan":
    daily_indexed = filtered.set_index("Time")
    data = resample_to_monthly(daily_indexed, drop_partial=False)
    data = data.reset_index().rename(columns={"Time": "Period"})
    period_col = "Period"
else:
    data = filtered
    period_col = "Time"

# Statistik
st.subheader("Statistik Data")
scol1, scol2, scol3, scol4 = st.columns(4)
with scol1:
    st.metric("Total Records", f"{len(data):,}")
with scol2:
    st.metric("Frekuensi", freq)
with scol3:
    st.metric(
        "Low Min", f"Rp {data['Low'].min():,.0f}" if not data.empty else "-"
    )
with scol4:
    st.metric(
        "Low Max", f"Rp {data['Low'].max():,.0f}" if not data.empty else "-"
    )

# Chart
st.subheader("Chart Harga")
chart_col = st.selectbox("Kolom", ["Low", "Close", "Open", "High"], index=0)
if not data.empty:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data[period_col], data[chart_col] / 1e9, color="#2196F3", linewidth=1)
    ax.set_xlabel("Periode")
    ax.set_ylabel(f"{chart_col} (Miliar IDR)")
    ax.set_title(f"BTC/IDR {chart_col} Price ({freq})")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Tabel
st.subheader("Tabel Data")
display = data.copy()
display[period_col] = pd.to_datetime(display[period_col]).dt.strftime("%Y-%m-%d")
for col in ["Open", "High", "Low", "Close"]:
    if col in display.columns:
        display[col] = display[col].apply(lambda x: f"Rp {x:,.0f}")
if "Volume" in display.columns:
    display["Volume"] = display["Volume"].apply(lambda x: f"{x:,.0f}")
st.dataframe(display, use_container_width=True, hide_index=True, height=400)

# ADF test bulanan (engine ARDL-ECM)
st.divider()
st.subheader("ADF Stationarity Test (bulanan)")
st.caption("Uji stasioneritas pakai engine ARDL-ECM pada data bulanan hasil resample.")
if st.button("Jalankan ADF Test"):
    daily_indexed = filtered.set_index("Time")
    monthly = resample_to_monthly(daily_indexed, drop_partial=True)
    if len(monthly) < 12:
        st.warning("Data bulanan terlalu sedikit untuk ADF test (min 12 bulan).")
    else:
        endog, exog = prepare_variables(monthly, log_transform=True)
        results = check_stationarity(endog, exog)
        adf_rows = []
        for name, r in results.items():
            if not isinstance(r, dict) or "adf_stat" not in r:
                continue
            order = r.get("order")
            status = {0: "I(0) Stasioner", 1: "I(1)", 2: "I(2) WARNING"}.get(
                order, "-"
            )
            adf_rows.append(
                {
                    "Variabel": name,
                    "ADF Statistic": f"{r['adf_stat']:.4f}",
                    "p-value": f"{r['p_value']:.4f}",
                    "Order": status,
                }
            )
        st.dataframe(
            pd.DataFrame(adf_rows), use_container_width=True, hide_index=True
        )
        if results.get("_any_i2"):
            st.error("Ada variabel I(2) — ARDL hanya valid untuk campuran I(0)/I(1).")
        else:
            st.success("Semua variabel I(0)/I(1) — ARDL valid.")

session.close()
