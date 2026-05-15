"""
BTC/IDR Price Prediction - Streamlit App
Entry point untuk multi-page Streamlit app (engine ARDL-ECM monthly)
"""
import streamlit as st

from db.engine import init_db, get_session
from db.repository import get_latest_run

# Inisialisasi database
init_db()

st.set_page_config(
    page_title="BTC/IDR Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar - info ringkas
with st.sidebar:
    st.title("BTC/IDR Prediction")
    st.caption("ARDL-ECM Model (Monthly, Econometric)")

    try:
        session = get_session()
        run_info, pred_df = get_latest_run(session)
        session.close()

        if run_info:
            st.divider()
            st.markdown("**Prediksi Terakhir**")
            st.write(
                f"Tanggal run: "
                f"{run_info['run_at'].strftime('%Y-%m-%d %H:%M') if run_info['run_at'] else '-'}"
            )
            st.write(f"Mode: {run_info['mode']}")
            if run_info.get("endog_lag") is not None:
                st.write(
                    f"Model: ARDL(p={run_info['endog_lag']}), "
                    f"{run_info.get('cointegration', '-')}"
                )
            if run_info.get("eval_mape") is not None:
                st.write(f"MAPE: {run_info['eval_mape']:.2f}%")
    except Exception:
        pass

# Landing page
st.title("Prediksi Harga Terendah Bulanan BTC/IDR")
st.markdown(
    """
Aplikasi prediksi harga **terendah bulanan** Bitcoin (BTC) dalam Rupiah Indonesia
(IDR) menggunakan model **ARDL-ECM** (Autoregressive Distributed Lags - Error
Correction Model) yang diestimasi secara ekonometrik.

### Metodologi
Data harian dari Indodax di-resample ke bulanan, lalu diproses lewat pipeline
ekonometrik proper:
- **ADF test** — uji stasioneritas (I(0)/I(1))
- **Lag selection** — pemilihan lag optimal via information criterion (AIC)
- **ARDL estimation** — estimasi OLS via `statsmodels`
- **Bounds test** — uji kointegrasi Pesaran-Shin-Smith
- **UECM** — Error Correction Model, speed-of-adjustment (λ)
- **VAR projection** — proyeksi variabel exogen untuk forecast

### Fitur
- **Dashboard** - Ringkasan prediksi terbaru
- **Prediksi Baru** - Jalankan prediksi: mode Forecast, Forecast+Evaluasi, atau Backtest
- **Riwayat Prediksi** - Browse semua histori run + config dari database
- **Evaluasi** - Bandingkan prediksi vs data aktual
- **Data Historis** - Eksplorasi data harga harian & bulanan

### Cara Penggunaan
1. Buka halaman **Prediksi Baru**, pilih mode, lalu jalankan prediksi
2. Hasil + config run tersimpan otomatis ke database
3. Lihat hasil di **Dashboard** atau **Riwayat Prediksi**
4. Gunakan **Evaluasi** untuk mengukur akurasi prediksi vs aktual

Gunakan menu di sidebar untuk navigasi antar halaman.
"""
)

st.divider()
st.caption(
    "Model ARDL-ECM monthly - Prediksi harga terendah bulanan BTC/IDR "
    "menggunakan data historis dari Indodax"
)
