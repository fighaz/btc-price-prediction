# BTC/IDR Price Prediction

Prediksi harga terendah harian Bitcoin (BTC) dalam Rupiah Indonesia (IDR) menggunakan model **ARDL (Autoregressive Distributed Lags)** dengan antarmuka web Streamlit.

## Fitur

- Prediksi harga terendah BTC/IDR menggunakan model ARDL
- Recursive forecasting N hari ke depan tanpa data aktual
- Confidence interval yang melebar seiring horizon prediksi
- Evaluasi otomatis prediksi vs data aktual
- Database SQLite untuk menyimpan histori prediksi
- Web UI menggunakan Streamlit (multi-page app)

## Struktur Folder

```
btc-price-prediction/
├── app.py                  # Entry point Streamlit
├── config.py               # Konfigurasi terpusat
├── requirements.txt        # Dependencies
├── src/                    # Core logic
│   ├── data.py             # Data acquisition dari Indodax API
│   ├── features.py         # Feature engineering ARDL
│   ├── statistics.py       # ADF stationarity test
│   ├── optimization.py     # AIC lag selection
│   ├── model.py            # ARDLModel class, training, forecasting
│   ├── evaluation.py       # Evaluasi forecast vs aktual
│   ├── visualization.py    # Chart generation
│   └── pipeline.py         # Orchestrator pipeline
├── db/                     # Database layer (SQLAlchemy)
│   ├── engine.py           # Engine & session factory
│   ├── models.py           # ORM table definitions
│   └── repository.py       # CRUD operations
├── pages/                  # Streamlit pages
│   ├── 1_Dashboard.py
│   ├── 2_Prediksi_Baru.py
│   ├── 3_Riwayat_Prediksi.py
│   ├── 4_Evaluasi.py
│   └── 5_Data_Historis.py
├── data/                   # SQLite database
├── output/                 # File output (CSV, PNG)
├── legacy/                 # Script original
└── docs/                   # Dokumentasi metodologi
```

## Instalasi

```bash
pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

## Sumber Data

Data harga BTC/IDR diambil dari [Indodax API](https://indodax.com) (timeframe harian, mulai 2014).

## Model

ARDL(p, q) dengan Ridge regularization:
- **AR component**: Lag harga Low (target)
- **DL component**: Lag harga Open, High, Close, dan Volume
- **Feature engineering**: Momentum, Moving Average, EMA, Volatility
- **Optimasi**: AIC grid search untuk lag order, CV untuk alpha
- **Anti-overfitting**: Minimum lag=2, MA/EMA dari Close (bukan target)
