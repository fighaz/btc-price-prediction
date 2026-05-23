"""
Simulasi DCA — pilih run backtest, bandingkan 4 strategi DCA di rentang bulan
yang identik sehingga perbandingan selalu adil (apel ke apel).
"""
import pandas as pd
import streamlit as st

from db.engine import get_session, init_db
from db.repository import (
    get_model_runs,
    get_predictions_for_run,
    get_price_history,
)
from src.ardl_ecm.data import resample_to_monthly
from src.ardl_ecm.dca import (
    simulate_dca,
    summary_dca,
    build_long_df,
    STRATEGY_LABELS,
)
from src.ardl_ecm.charts import dca_price_comparison_chart

st.set_page_config(page_title="Simulasi DCA", page_icon="💰", layout="wide")
st.title("💰 Simulasi DCA")
st.caption(
    "Pilih hasil backtest model ARDL-ECM, lalu bandingkan 4 strategi DCA "
    "di rentang bulan yang sama."
)

init_db()
session = get_session()

# ============================================================================
# Ambil run backtest yang tersedia
# ============================================================================
all_runs = get_model_runs(session, limit=50)
bt_runs = [
    r for r in all_runs
    if r["mode"] == "backtest" and r["status"] == "success"
]

if not bt_runs:
    st.warning(
        "Belum ada run **Backtest** yang sukses. "
        "Buka **🔮 Prediksi Baru**, pilih mode *Backtest (walk-forward)*, "
        "lalu jalankan prediksi terlebih dahulu."
    )
    session.close()
    st.stop()

price_df = get_price_history(session)
if price_df.empty:
    st.warning(
        "Data harga historis belum tersedia. Jalankan **🔮 Prediksi Baru** "
        "untuk mengisi data harga ke database."
    )
    session.close()
    st.stop()

# ============================================================================
# Form: pilih run backtest + modal
# ============================================================================
def _run_label(r):
    return (
        f"Run #{r['id']} — {r['backtest_months']} bulan "
        f"(train s/d {r['train_end_date']})"
    )

with st.form("dca_form"):
    col1, col2 = st.columns(2)
    with col1:
        selected_run_id = st.selectbox(
            "Pilih run backtest",
            options=[r["id"] for r in bt_runs],
            format_func=lambda rid: _run_label(
                next(r for r in bt_runs if r["id"] == rid)
            ),
            help="Rentang bulan simulasi DCA akan mengikuti bulan-bulan "
                 "yang ada di run backtest ini.",
        )
        modal = st.number_input(
            "Modal per bulan (Rp)",
            min_value=10_000,
            max_value=100_000_000,
            value=500_000,
            step=50_000,
            format="%d",
        )
    with col2:
        st.markdown("**Tentang strategi:**")
        st.markdown(
            "- **Awal Bulan** — beli di harga Open hari pertama\n"
            "- **Akhir Bulan** — beli di harga Close hari terakhir\n"
            "- **Low Bulanan** — beli di harga terendah *(ideal / upper bound)*\n"
            "- **Prediksi Model** — beli di harga prediksi ARDL-ECM; "
            "model dijalankan di awal bulan menggunakan Open hari pertama "
            "sebagai sinyal, memprediksi Low bulan yang sama; "
            "jika prediksi < Low aktual *(miss)*, fallback ke Close bulan itu "
            "agar total investasi identik antar strategi *(fair comparison)*"
        )

    submitted = st.form_submit_button(
        "🚀 Jalankan Simulasi", type="primary", use_container_width=True
    )

if not submitted:
    session.close()
    st.stop()

# ============================================================================
# Persiapan data
# ============================================================================
with st.spinner("Memuat data dan menjalankan simulasi..."):
    selected_run = next(r for r in bt_runs if r["id"] == selected_run_id)
    pred_df_raw = get_predictions_for_run(session, selected_run_id)

    if pred_df_raw.empty:
        st.error("Run ini tidak punya data prediksi.")
        session.close()
        st.stop()

    pred_df_raw["Target_Month"] = pd.to_datetime(pred_df_raw["Target_Month"])
    bulan_backtest = sorted(
        pred_df_raw["Target_Month"].dt.strftime("%Y-%m").unique()
    )
    bulan_start = bulan_backtest[0]
    bulan_end   = bulan_backtest[-1]

    # Filter price_history ke rentang bulan backtest
    ts_start = pd.Timestamp(bulan_start + "-01")
    ts_end   = pd.Timestamp(bulan_end + "-01") + pd.offsets.MonthEnd(1)
    daily_range = price_df[
        (price_df["Time"] >= ts_start) & (price_df["Time"] <= ts_end)
    ].copy()

    if len(daily_range) < 20:
        st.error("Data harian untuk rentang ini terlalu sedikit.")
        session.close()
        st.stop()

    # Resample ke bulanan, filter ke bulan-bulan backtest saja
    monthly_full = resample_to_monthly(daily_range.set_index("Time"), drop_partial=False)
    monthly = monthly_full[
        monthly_full.index.strftime("%Y-%m").isin(bulan_backtest)
    ]

    if len(monthly) < 2:
        st.error("Butuh minimal 2 bulan data.")
        session.close()
        st.stop()

    # pred_df sudah pasti dalam rentang yang sama → align_to_model=True otomatis
    dca_result = simulate_dca(
        monthly_df=monthly,
        daily_df=daily_range,
        modal_per_bulan=float(modal),
        pred_df=pred_df_raw,
        align_to_model=True,   # selalu adil: semua beli di bulan yang sama
    )

    if not dca_result:
        st.error("Simulasi gagal — tidak ada data yang cukup.")
        session.close()
        st.stop()

    harga_akhir = float(monthly["Close"].iloc[-1])
    akhir_bulan = monthly.index[-1].strftime("%Y-%m")
    summary     = summary_dca(dca_result, harga_akhir)
    long_df     = build_long_df(dca_result)

# ============================================================================
# Header info
# ============================================================================
st.subheader(f"📊 Hasil Simulasi — Run #{selected_run_id}")
st.caption(
    f"Rentang: **{bulan_start}** – **{akhir_bulan}** "
    f"({len(monthly)} bulan) · "
    f"Modal: **Rp {modal:,.0f}/bln** · "
    f"Total investasi: **Rp {modal * len(monthly):,.0f}** · "
    f"Harga penutupan akhir: **Rp {harga_akhir:,.0f}**"
)

# ============================================================================
# Kartu ringkasan per strategi
# ============================================================================
st.caption(
    f"Return % = (Total BTC × harga penutupan akhir − total investasi) ÷ total investasi × 100. "
    f"Harga penutupan akhir periode: **Rp {harga_akhir:,.0f}**"
)
cols = st.columns(len(summary))
for col, (_, row) in zip(cols, summary.iterrows()):
    ret = row["Return_Pct"]
    col.metric(
        label=row["Label"],
        value=f"Rp {row['Average_Cost']/1e9:,.4f}M",
        delta=f"Return {ret:+.1f}%",
        delta_color="normal",
        help=(
            f"Total BTC      : {row['Total_BTC']:.8f}\n"
            f"Avg cost       : Rp {row['Average_Cost']:,.0f}\n"
            f"Total investasi: Rp {row['Total_Invested']:,.0f}\n"
            f"Nilai portofolio: Rp {row['Portfolio_Value']:,.0f}\n"
            f"Return = (Rp {row['Portfolio_Value']:,.0f} - Rp {row['Total_Invested']:,.0f}) "
            f"/ Rp {row['Total_Invested']:,.0f} × 100 = {ret:+.2f}%"
        ),
    )

# Ringkasan miss/hit strategi model
if "model" in dca_result and "Status" in dca_result["model"].columns:
    model_summary = summary[summary["Strategi"] == "model"]
    if not model_summary.empty:
        ms = model_summary.iloc[0]
        n_hit  = int(ms["Hit_Count"])  if ms["Hit_Count"]  is not None else 0
        n_miss = int(ms["Miss_Count"]) if ms["Miss_Count"] is not None else 0
        hit_rate = ms["Hit_Rate_Pct"]
        n_skip = int(dca_result["model"]["Status"].str.startswith("Tidak").sum())
        st.caption(
            f"Prediksi Model: **{n_hit} bulan hit** (beli di predicted low) · "
            f"**{n_miss} bulan miss** → fallback ke Close · "
            f"**{n_skip} bulan tanpa prediksi** · "
            f"Hit Rate: **{hit_rate:.1f}%**"
        )

st.divider()

# ============================================================================
# Charts
# ============================================================================
st.subheader("📊 Perbandingan Harga Beli per Bulan")
st.altair_chart(dca_price_comparison_chart(long_df), use_container_width=True)
st.caption(
    "Harga beli tiap bulan per strategi. Prediksi Model: titik solid = berhasil beli, "
    "titik transparan = miss (harga aktual lebih tinggi, fallback ke Close). "
    "Arahkan kursor untuk detail; scroll untuk zoom."
)

st.divider()

# ============================================================================
# Tabel harga beli per bulan
# ============================================================================
session.close()

st.subheader("📅 Harga Beli per Bulan")

strategy_keys = list(dca_result.keys())
all_months = sorted({b for s in strategy_keys for b in dca_result[s]["Bulan"]})
harga_tbl = pd.DataFrame({"Bulan": all_months})

for s in strategy_keys:
    df_s = dca_result[s].copy()
    if s == "model":
        df_s = df_s.rename(columns={
            "Harga_Prediksi": "Harga_Prediksi_Model",
            "Harga_Beli":     "Harga_Eksekusi_Model",
            "Status":         "Status_Model",
        })
        pick = ["Bulan", "Harga_Prediksi_Model", "Harga_Eksekusi_Model", "Status_Model"]
    else:
        df_s = df_s.rename(columns={"Harga_Beli": f"Harga_{s}"})
        pick = ["Bulan", f"Harga_{s}"]
    harga_tbl = harga_tbl.merge(df_s[pick], on="Bulan", how="left")

harga_tbl = harga_tbl.sort_values("Bulan").reset_index(drop=True)

cfg = {"Bulan": st.column_config.TextColumn("Bulan")}
for s in strategy_keys:
    lbl = STRATEGY_LABELS.get(s, s)
    if s == "model":
        cfg["Harga_Prediksi_Model"] = st.column_config.NumberColumn(
            "Prediksi Model", format="Rp %.0f"
        )
        cfg["Harga_Eksekusi_Model"] = st.column_config.NumberColumn(
            "Eksekusi Model", format="Rp %.0f"
        )
        cfg["Status_Model"] = st.column_config.TextColumn("Status")
    else:
        cfg[f"Harga_{s}"] = st.column_config.NumberColumn(lbl, format="Rp %.0f")

st.caption(
    "**Prediksi Model** = harga prediksi ARDL-ECM di awal bulan. "
    "**Eksekusi Model** = harga yang benar-benar dipakai beli "
    "(sama saat Hit, fallback ke Close saat Miss)."
)
st.dataframe(harga_tbl, use_container_width=True, hide_index=True, column_config=cfg)

csv = harga_tbl.to_csv(index=False)
st.download_button(
    "⬇️ Download CSV",
    csv,
    file_name=f"harga_beli_run{selected_run_id}_{bulan_start}_{akhir_bulan}.csv",
    mime="text/csv",
)
