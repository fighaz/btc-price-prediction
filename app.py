"""
BTC/IDR Price Prediction - Streamlit App
Entry point untuk multi-page Streamlit app
"""
import streamlit as st
from db.engine import init_db, get_session
from db.repository import get_latest_prediction, get_model_runs

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
    st.caption("ARDL Model (Autoregressive Distributed Lags)")

    try:
        session = get_session()
        run_info, pred_df = get_latest_prediction(session)
        session.close()

        if run_info:
            st.divider()
            st.markdown("**Prediksi Terakhir**")
            st.write(f"Tanggal run: {run_info['run_at'].strftime('%Y-%m-%d %H:%M') if run_info['run_at'] else '-'}")
            st.write(f"Model: ARDL(p={run_info['p_lags']}, q={run_info['q_lags']})")
            st.write(f"R²: {run_info['train_r2']:.4f}" if run_info['train_r2'] else "")
            if run_info.get("eval_mape"):
                st.write(f"MAPE: {run_info['eval_mape']:.2f}%")
    except Exception:
        pass

# Landing page
st.title("Prediksi Harga Terendah BTC/IDR")
st.markdown("""
Aplikasi prediksi harga terendah harian Bitcoin (BTC) dalam Rupiah Indonesia (IDR)
menggunakan model **ARDL (Autoregressive Distributed Lags)**.

### Fitur
- **Dashboard** - Ringkasan prediksi terbaru dengan chart interaktif
- **Prediksi Baru** - Jalankan prediksi baru dengan parameter custom
- **Riwayat Prediksi** - Browse semua histori prediksi dari database
- **Evaluasi** - Bandingkan prediksi vs data aktual
- **Data Historis** - Eksplorasi data harga mentah BTC/IDR

### Cara Penggunaan
1. Buka halaman **Prediksi Baru** untuk menjalankan prediksi pertama
2. Hasil akan tersimpan otomatis ke database
3. Lihat hasil di **Dashboard** atau **Riwayat Prediksi**
4. Setelah periode prediksi berlalu, gunakan **Evaluasi** untuk mengukur akurasi

Gunakan menu di sidebar untuk navigasi antar halaman.
""")

st.divider()
st.caption("Model ARDL - Prediksi harga terendah BTC/IDR menggunakan data historis dari Indodax")
