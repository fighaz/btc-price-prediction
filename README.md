# Prediksi Harga Low BTC/IDR — ARDL-ECM Monthly

Aplikasi prediksi harga terendah bulanan Bitcoin/Rupiah menggunakan model **ARDL-ECM (Autoregressive Distributed Lag – Error Correction Model)** berbasis data Indodax. Dirancang dengan asumsi eksekusi di awal bulan: model dijalankan pada tanggal 1, menggunakan harga Open hari pertama sebagai sinyal tambahan, lalu memprediksi harga Low bulan yang sedang berjalan.

---

## Struktur Folder

```
btc-price-prediction/
├── app.py                        # Entry point Streamlit
├── requirements.txt
├── src/
│   ├── pipeline.py               # Orchestrator: forecast, backtest
│   └── ardl_ecm/
│       ├── config.py             # Konstanta & parameter default
│       ├── data.py               # Fetch API + resample harian→bulanan
│       ├── model.py              # ARDL, bounds test, ECM
│       ├── forecast.py           # VAR exog projection + monthly forecast
│       ├── backtest.py           # Walk-forward backtest
│       ├── dca.py                # Simulasi Dollar-Cost Averaging
│       └── charts.py             # Visualisasi Altair
├── db/
│   ├── engine.py                 # SQLAlchemy engine & session
│   ├── models.py                 # ORM table definitions
│   └── repository.py             # CRUD operations
└── pages/
    ├── 1_Dashboard.py
    ├── 2_Prediksi_Baru.py
    ├── 3_Riwayat_Prediksi.py
    ├── 4_Evaluasi.py
    ├── 5_Data_Historis.py
    └── 6_Simulasi_DCA.py
```

---

## Instalasi & Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Parameter & Konfigurasi Default

Semua parameter ada di `src/ardl_ecm/config.py` dan bisa di-override per run dari UI.

| Parameter | Default | Keterangan |
|---|---|---|
| `MAX_LAG_ENDOG` | 4 | Maksimum lag untuk variabel Low (target) |
| `MAX_LAG_EXOG` | 4 | Maksimum lag untuk Open, Close, Volume |
| `VAR_MAXLAG` | 4 | Maksimum lag untuk model VAR proyeksi exog |
| `IC` | `"aic"` | Information criterion pemilihan lag (AIC) |
| `TREND` | `"c"` | Spesifikasi trend: konstanta saja |
| `USE_LOG_TRANSFORM` | `True` | Log-transform OHLCV untuk stabilisasi variansi |
| `ROLLING_WINDOW_YEARS` | `0` | `0` = gunakan seluruh data sejak 2014 |
| `BACKTEST_MONTHS` | `12` | Jumlah bulan walk-forward backtest |
| `MIN_DAYS_PER_MONTH` | `20` | Bulan dengan data < 20 hari di-drop |

---

## Alur Lengkap: Dari Pengambilan Data Hingga Output Prediksi

### Asumsi Utama

> Model selalu dijalankan pada **tanggal 1 bulan M**. Data training mencakup seluruh data historis s/d akhir bulan M-1. Harga **Open hari pertama bulan M** diambil dari API dan diinjeksikan sebagai sinyal nyata ke dalam model, menggantikan proyeksi VAR untuk variabel Open.

---

### Langkah 1 — Pengambilan Data Harian (`data.py`)

**Fungsi:** `fetch_btc_daily(to_date)`

Data harian BTC/IDR diambil dari Indodax API dengan parameter:
- `from`: epoch timestamp awal (sejak 2014)
- `to`: epoch timestamp dari `to_date` (= tanggal 1 bulan M)
- `symbol`: pasangan trading BTC/IDR
- `tf`: timeframe harian (`1D`)

Setelah data diterima:
1. Kolom `Time` dikonversi dari Unix epoch ke datetime, dinormalisasi ke tanggal (jam dihapus)
2. Kolom OHLCV di-cast ke float
3. Baris duplikat dihapus, diurutkan ascending
4. Nilai `0` pada OHLCV diganti NaN → forward-fill → interpolasi linear
5. Index diset ke datetime dengan frekuensi harian (`D`), gap hari libur di-forward-fill

**Output:** DataFrame harian dengan index DatetimeIndex (freq `D`), kolom `[Open, High, Low, Close, Volume]`

---

### Langkah 2 — Resample ke Data Bulanan (`data.py`)

**Fungsi:** `resample_to_monthly(df_daily, drop_partial, min_days=20)`

Data harian diagregasi per bulan kalender dengan aturan:

| Kolom | Agregasi | Keterangan |
|---|---|---|
| `Open` | `.first()` | Harga pembukaan hari pertama bulan |
| `High` | `.max()` | Harga tertinggi sepanjang bulan |
| `Low` | `.min()` | **Target prediksi Y** — harga terendah bulan |
| `Close` | `.last()` | Harga penutupan hari terakhir bulan |
| `Volume` | `.sum()` | Total volume bulan |
| `log_Volume` | `log1p(Volume)` | Dihitung setelah agregasi |

**Drop partial:**
- `drop_partial=True` (default untuk training): bulan dengan data < 20 hari di-drop. Memastikan bulan yang sedang berjalan tidak masuk data training.
- `drop_partial=False` (untuk backtest & evaluasi): bulan partial tetap disertakan agar `actual_row` tersedia.

---

### Langkah 3 — Pemisahan Variabel & Log-Transform (`model.py`)

**Fungsi:** `prepare_variables(monthly, log_transform=True)`

Data bulanan dipisah menjadi:
- **Endogen (Y):** `log(Low)` — target yang diprediksi
- **Exogen (X):** `[log_Open, log_Close, log_Volume]`

Log-transform diterapkan untuk:
1. Menstabilkan variansi (heteroskedasticity)
2. Membuat distribusi residual lebih mendekati normal
3. Menginterpretasikan koefisien sebagai elastisitas

---

### Langkah 4 — Uji Stasioneritas ADF (`model.py`)

**Fungsi:** `check_stationarity(endog, exog)`

Uji Augmented Dickey-Fuller dijalankan untuk setiap variabel (`Low`, `log_Open`, `log_Close`, `log_Volume`):
1. **Level test:** apakah variabel stasioner di level (I(0))?
2. Jika tidak stasioner (p ≥ 0.05): uji first-difference → I(1) atau I(2)

ARDL dapat menangani campuran I(0) dan I(1); variabel I(2) akan menghasilkan peringatan.

---

### Langkah 5 — Pemilihan Lag Optimal (`model.py`)

**Fungsi:** `select_lag_order(endog, exog, max_lag_endog=4, max_lag_exog=4, ic="aic")`

Menggunakan `ardl_select_order()` dari statsmodels untuk mencari kombinasi lag yang meminimalkan AIC:
- `endog_lag` (p): jumlah lag variabel Low yang masuk model, maksimum 4
- `exog_orders` (q): jumlah lag per variabel exogen, maksimum 4 masing-masing

Contoh hasil: `endog_lag=2, exog_orders={'log_Open': 1, 'log_Close': 1, 'log_Volume': 0}`

---

### Langkah 6 — Estimasi Model ARDL (`model.py`)

**Fungsi:** `estimate_ardl(endog, exog, endog_lag, exog_orders, trend="c")`

Model ARDL(p, q₁, q₂, q₃) diestimasi via OLS menggunakan statsmodels:

```
log(Low)_t = c
           + Σᵢ αᵢ · log(Low)_{t-i}       (i = 1..p)
           + Σⱼ β₁ⱼ · log(Open)_{t-j}     (j = 0..q₁)
           + Σⱼ β₂ⱼ · log(Close)_{t-j}    (j = 0..q₂)
           + Σⱼ β₃ⱼ · log(Volume)_{t-j}   (j = 0..q₃)
           + εₜ
```

Setelah estimasi, dihitung R² in-sample sebagai baseline goodness-of-fit.

---

### Langkah 7 — Bounds Test Kointegrasi (Pesaran-Shin-Smith) (`model.py`)

**Fungsi:** `run_bounds_test(endog, exog, endog_lag, exog_orders, trend="c")`

Uji kointegrasi untuk memastikan ada hubungan jangka panjang antara Low dengan Open/Close/Volume:

1. Estimasi UECM (Unrestricted ECM) dengan syarat lag minimum 1
2. Hitung F-statistic dari joint hypothesis test
3. Bandingkan dengan critical bounds pada signifikansi 5%:
   - F > batas atas → **COINTEGRATED** — ada hubungan jangka panjang
   - F < batas bawah → **NOT_COINTEGRATED** — tidak ada
   - Di antara → **INCONCLUSIVE** — diasumsikan cointegrated

---

### Langkah 8 — Interpretasi ECM (`model.py`)

**Fungsi:** `interpret_ecm(uecm_fit, cointegrated)`

Jika cointegrated, diekstrak:
- **λ (speed of adjustment):** koefisien lagged level term, bernilai negatif. Menunjukkan seberapa cepat deviasi dari keseimbangan jangka panjang terkoreksi per bulan.
- **Half-life:** `log(0.5) / log(1 + λ)` — berapa bulan hingga deviasi berkurang 50%
- **Koefisien jangka panjang** per variabel exogen
- **Diagnostik residual:**
  - Breusch-Godfrey LM test (lag 4) — deteksi serial correlation
  - ARCH LM test (lag 4) — deteksi heteroskedasticity
  - Jarque-Bera — uji normalitas residual
  - CUSUM — uji stabilitas struktural

---

### Langkah 9 — Proyeksi Variabel Exogen via VAR (`forecast.py`)

**Fungsi:** `forecast_exog_var(exog, horizon=1, var_maxlag=4)`

Karena nilai Open/Close/Volume bulan M belum diketahui sepenuhnya, ketiga variabel ini diproyeksikan 1 bulan ke depan menggunakan model VAR(p):
1. Fit VAR ke seluruh data exog historis, lag dipilih via AIC (maks 4)
2. Gunakan p observasi terakhir sebagai seed
3. Forecast 1 langkah ke depan

**Output:** DataFrame 1 baris dengan kolom `[log_Open, log_Close, log_Volume]` untuk bulan M.

---

### Langkah 10 — Injeksi Open Aktual Hari Pertama (`pipeline.py`)

Setelah proyeksi VAR selesai, **nilai `log_Open` diganti** dengan harga Open hari pertama bulan M yang sudah diketahui secara nyata (di-fetch dari API pada Langkah 1):

```python
exog_future["log_Open"] = np.log(current_open)
```

Ini adalah inti dari asumsi "dijalankan di awal bulan" — kita tidak perlu menebak Open bulan ini karena sudah terjadi di tanggal 1.

---

### Langkah 11 — Forecast Monthly Low (`forecast.py`)

**Fungsi:** `forecast_monthly(fit, endog, exog_future, horizon=1, log_transform=True)`

Prediksi 1 langkah ke depan menggunakan `fit.get_prediction()` dengan `exog_oos = exog_future`:
1. Dapatkan `ŷ` (prediksi log-Low) beserta 95% Confidence Interval
2. Hitung bias correction: `σ² = var(residuals)`
3. Inverse transform ke skala Rupiah:
   - `Predicted_Low = exp(ŷ + σ²/2)`
   - `CI_Lower = exp(ci_lower + σ²/2)`
   - `CI_Upper = exp(ci_upper + σ²/2)`

Faktor `σ²/2` mencegah underestimasi sistematis akibat Jensen's inequality saat inverse log-transform.

**Output:** DataFrame 1 baris:

| Kolom | Contoh |
|---|---|
| `Predicted_Low` | Rp 1.560.000.000 |
| `CI_Lower` | Rp 1.480.000.000 |
| `CI_Upper` | Rp 1.640.000.000 |

---

## Alur Walk-Forward Backtest

Backtest mensimulasikan bagaimana model bekerja di masa lalu secara realistis — setiap iterasi menggunakan hanya data yang tersedia pada saat itu.

**Fungsi:** `rolling_backtest(monthly, n_backtest_months=12, ...)`

```
Data bulanan total: N bulan

Iterasi i = 12, 11, ..., 1:
  ┌─────────────────────────────────────────────────────┐
  │ Training data : bulan 1 s/d (N - i)                 │
  │ Target bulan  : bulan (N - i + 1)  ← actual_row    │
  │                                                      │
  │ 1. Lag selection ulang (AIC)                         │
  │ 2. ARDL estimation ulang                             │
  │ 3. Bounds test                                       │
  │ 4. VAR forecast exog                                 │
  │ 5. Injeksi actual_row["Open"] ke exog_future        │
  │ 6. Forecast Low bulan target                         │
  │ 7. Bandingkan vs actual_row["Low"]:                  │
  │    error_pct = |actual - pred| / actual × 100        │
  │    in_ci     = CI_Lower ≤ actual ≤ CI_Upper          │
  └─────────────────────────────────────────────────────┘
```

**Catatan penting:** `actual_row["Open"]` diinjeksikan (bukan Low/Close) — konsisten dengan asumsi bahwa di awal bulan M kita hanya tahu harga pembukaan hari pertama, bukan harga terendah atau penutupan bulan itu.

**Metrik agregat hasil backtest:**

| Metrik | Keterangan |
|---|---|
| MAPE | Mean Absolute Percentage Error (%) |
| RMSE | Root Mean Squared Error (Rp) |
| MAE | Mean Absolute Error (Rp) |
| R² | Koefisien determinasi |
| CI Coverage | % bulan di mana aktual jatuh dalam interval 95% |
| Cointegration Rate | % iterasi yang menghasilkan kointegrasi |

---

## Simulasi Dollar-Cost Averaging (DCA)

Setelah backtest menghasilkan prediksi per bulan, halaman Simulasi DCA membandingkan 4 strategi investasi dengan modal tetap setiap bulan.

**Fungsi:** `simulate_dca(monthly_df, modal_per_bulan, pred_df, align_to_model=True)`

| Strategi | Harga Beli | Keterangan |
|---|---|---|
| Awal Bulan (Open) | Open hari pertama | Beli langsung di awal bulan |
| Akhir Bulan (Close) | Close hari terakhir | Beli di akhir bulan |
| Low Bulanan (Ideal) | Low terendah bulan | Timing sempurna, benchmark ideal |
| Prediksi Model | Predicted_Low ARDL-ECM | Berdasarkan prediksi model |

**Logika Hit/Miss untuk strategi model:**
- **Hit** (`predicted_low ≥ low_aktual`): Harga aktual turun ke level prediksi atau lebih rendah → beli di harga prediksi
- **Miss** (`predicted_low < low_aktual`): Harga aktual tidak pernah serendah prediksi → fallback ke harga Close bulan itu agar total investasi identik antar strategi (perbandingan fair)

**Metrik output per strategi:**
- Total BTC terkumpul
- Total investasi (Rp)
- Average cost per BTC (Rp)
- Nilai portofolio di harga penutupan akhir periode
- Return % = (nilai portofolio − total investasi) / total investasi × 100

---

## Ringkasan Alur Visual

```
Tanggal 1 bulan M
        │
        ▼
┌───────────────────┐
│ 1. Fetch API      │  Data harian BTC/IDR s/d tanggal 1 bulan M
│    Indodax        │  → ambil Open hari pertama bulan M
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 2. Resample       │  Harian → Bulanan (drop_partial=True)
│    Bulanan        │  Low bulanan = target Y
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 3. ADF Test       │  Uji stasioneritas semua variabel
│    Stasioneritas  │  I(0) / I(1) per variabel
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 4. Lag Selection  │  ardl_select_order() dengan AIC
│    AIC            │  → endog_lag, exog_orders
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 5. ARDL Estimation│  OLS → fit, koefisien, residual
│    + Bounds Test  │  Pesaran-Shin-Smith kointegrasi
│    + ECM          │  λ, half-life, long-run coefs
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 6. VAR Projection │  Proyeksi log_Open, log_Close, log_Volume
│    Exog           │  1 bulan ke depan
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 7. Injeksi Open   │  Ganti log_Open proyeksi VAR
│    Aktual         │  dengan log(Open hari pertama bulan M)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 8. Forecast Low   │  ARDL 1-step ahead
│    Bulan M        │  + bias correction exp(σ²/2)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Output            │  Predicted_Low ± CI 95%  (dalam Rupiah)
└───────────────────┘
```

---

## Sumber Data

Data harga BTC/IDR diambil dari [Indodax](https://indodax.com) melalui API publik, timeframe harian, mulai tahun 2014 hingga tanggal fetch.
