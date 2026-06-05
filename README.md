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

## Tiga Jenis Lag — Jangan Tertukar

Pipeline ini punya **tiga konfigurasi "lag" yang berbeda** dan sering disalahpahami. Ketiganya bekerja pada model yang berbeda dan untuk tujuan yang berbeda.

| Konstanta | Milik model | Untuk apa | Hasil terpilih |
|---|---|---|---|
| `MAX_LAG_ENDOG` | **ARDL** (model utama prediksi Low) | Batas atas berapa bulan lalu nilai **Low sendiri** dipakai memprediksi Low | `endog_lag` (p) |
| `MAX_LAG_EXOG` | **ARDL** (model utama prediksi Low) | Batas atas berapa bulan lalu **Open/Close/Volume** dipakai memprediksi Low | `exog_orders` (q per variabel) |
| `VAR_MAXLAG` | **VAR** (sub-model proyeksi eksogen) | Batas atas berapa bulan lalu dipakai **memproyeksikan Open/Close/Volume bulan depan** | `k_ar` (lag VAR terpilih) |

### Kenapa VAR diperlukan sama sekali?

Model ARDL memprediksi `Low` bulan depan, tetapi sebagai input ia butuh nilai **eksogen bulan depan** (`log_Open`, `log_Close`, `log_Volume`) — yang belum terjadi. Maka eksogen masa depan harus **diproyeksikan dulu** dengan VAR (Langkah 9), baru hasilnya dimasukkan ke ARDL (Langkah 11). Khusus `log_Open`, proyeksi VAR diganti harga Open aktual hari pertama bulan target (Langkah 10).

```
MAX_LAG_ENDOG / MAX_LAG_EXOG ──► ARDL: prediksi Low dari masa lalu Low + eksogen
                                          ▲
                                          │ butuh eksogen BULAN DEPAN sebagai input
                                          │
VAR_MAXLAG ─────────────────────► VAR: proyeksikan eksogen bulan depan
```

> **Catatan:** semua nilai di atas hanyalah **batas atas pencarian**. Lag yang benar-benar dipakai dipilih otomatis via AIC dan biasanya lebih kecil dari batasnya.

### Arti angka order (mis. `q = 0` vs `q = 2`)

Untuk variabel **eksogen**, angka order = **lag maksimum** yang dipakai; variabel selalu disertakan dari lag 0 sampai angka itu:

| Notasi | Lag yang dipakai | Arti |
|---|---|---|
| `log_Open: 0` | lag 0 | Open **bulan berjalan** saja (kontemporer). Order 0 **bukan** berarti tidak dipakai. |
| `log_Close: 2` | lag 0, 1, 2 | Close bulan berjalan + 2 bulan sebelumnya |
| `log_Volume: 1` | lag 0, 1 | Volume bulan berjalan + 1 bulan sebelumnya |

Untuk variabel **endogen** (`p`) aturannya beda — **tidak** memakai lag 0 (karena lag 0 dari Low adalah nilai yang sedang diprediksi):

| Notasi | Lag yang dipakai |
|---|---|
| `p = 4` | Low pada lag 1, 2, 3, 4 |

Jadi `ARDL(p=4, orders={'log_Open': 0, 'log_Close': 2, 'log_Volume': 1})` berarti notasi **ARDL(4, 0, 2, 1)**.

---

## Analisis Pemilihan Max Lag

`max_lag_endog` dan `max_lag_exog` adalah **batas atas** pencarian lag (lihat "Tiga Jenis Lag"). Pertanyaannya: berapa batas yang sebaiknya dipakai? Dua perspektif diuji — fit *in-sample* (AIC) dan akurasi *out-of-sample* (backtest).

### Perspektif 1 — AIC (fit in-sample)

Grid-search `ardl_select_order` dijalankan pada data training (147 bulan, s/d April 2026) dengan max lag berbeda; target forecast Mei 2026. AIC mengukur trade-off goodness-of-fit vs kompleksitas (makin kecil makin baik):

| Max lag | Model terpilih (p, q-orders) | AIC | Prediksi Mei | Jumlah kandidat |
|---|---|---|---|---|
| 4 | ARDL(4, {Open:0, Close:2, Vol:1}) | -267.82 | Rp 1.199 M | 1.080 |
| **6** | ARDL(5, {Open:3, Close:0, Vol:6}) | **-270.32** | Rp 1.222 M | 3.584 |
| 8 | ARDL(5, {Open:3, Close:0, Vol:6}) | -270.32 | Rp 1.253 M | 9.000 |
| 12 | ARDL(5, {Open:3, Close:0, Vol:6}) | -270.32 | Rp 1.254 M | 35.672 |

- Menaikkan max lag 4 → 6 menurunkan AIC (-267.82 → -270.32): ada struktur lag yang lebih baik yang **terlewat** saat batas hanya 4.
- Max lag **8 dan 12 menghasilkan model identik** dengan 6 (AIC & orders sama) — pencarian sudah **konvergen di lag 6**. Memperbesar batas hanya menambah jumlah kandidat (3.584 → 35.672) dan memperlama komputasi tanpa menemukan model lebih baik. (Prediksi sedikit bergeser di lag 8/12 karena `var_maxlag` proyeksi VAR ikut membesar.)
- Aturan praktis: max lag wajar ≤ n/4 (untuk 147 bulan ≈ 37), tapi untuk data **bulanan** lag > 12 jarang bermakna secara ekonomis.
- Aktual Low Mei 2026 = **Rp 1.294 M** — semua konfigurasi memprediksi di kisaran wajar (error 3–7%).

### Perspektif 2 — Backtest walk-forward (akurasi out-of-sample)

Backtest dengan **horizon tetap 12 bulan**, max lag divariasikan (data s/d Mei 2026):

| Max lag | MAPE | RMSE | MAE | R² | CI Coverage | Coint |
|---|---|---|---|---|---|---|
| 3 | 8.39% | Rp 140.8 M | Rp 114.1 M | 0.7228 | 91.7% | 100% |
| **4** | **7.88%** | Rp 134.2 M | Rp 106.9 M | 0.7482 | 91.7% | 100% |
| 6 | 8.37% | Rp 134.3 M | Rp 112.6 M | 0.7479 | 83.3% | 100% |
| 12 | 8.25% | Rp 136.6 M | Rp 111.0 M | 0.7392 | 83.3% | 100% |

- **MAPE terbaik di max lag 4** (7.88%) — bukan 6. Max lag 3, 6, 12 semuanya lebih buruk (8.25–8.39%). Menambah max lag **tidak otomatis** memperbaiki akurasi out-of-sample.
- **Max lag 4** juga unggul di MAE (106.9 M) dan **CI coverage tertinggi** (91.7%). Max lag 6 sedikit lebih sempit intervalnya (CI coverage turun ke 83.3%) sehingga lebih sering meleset dari aktual.
- **Max lag 12** tidak memberi keunggulan meski paling kompleks — indikasi diminishing return / **over-fitting**: lag tambahan menangkap noise, bukan sinyal.
- Kointegrasi terdeteksi 100% di semua konfigurasi.

### Kesimpulan

AIC in-sample favor **max lag 6**, tetapi backtest out-of-sample favor **max lag 4** (MAPE, MAE, & CI coverage terbaik). Ini pelajaran penting: **AIC mengukur fit pada data latih, bukan akurasi prediksi masa depan** — keduanya bisa menunjuk arah berbeda. Karena prioritas model ini adalah akurasi prediksi nyata, **default dipertahankan di max lag 4**. Max lag tetap dapat diatur dari form Prediksi Baru bila ingin bereksperimen.

> Catatan: nilai akhir lag yang **dipakai** model selalu dipilih otomatis via AIC dalam batas yang ditentukan, dan biasanya lebih kecil dari batas itu.

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
