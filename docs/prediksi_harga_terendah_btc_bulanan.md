# Panduan Prediksi Harga Terendah Bitcoin Bulanan
### Berdasarkan: *Optimal Prediction of Bitcoin High and Low Prices: An Exploratory Analysis* — Tatiana Rice (2025)

---

## Ringkasan Model Terbaik

Model terbaik untuk memprediksi **harga terendah (low) Bitcoin bulanan** adalah:

> **ARDL Bull Monthly Model**
> - Rata-rata MAPE (out-of-sample): **3.28%**
> - Jauh di bawah batas akurasi tinggi Lewis (1982): **10%**

---

## Alur Proses Lengkap

```
[1. Kumpulkan Data]
        ↓
[2. Preprocessing Data]
        ↓
[3. Seleksi Variabel Independen]
        ↓
[4. Uji Stasioneritas (ADF Test)]
        ↓
[5. Analisis Correlogram]
        ↓
[6. Bangun Model ARDL Bull Monthly]
        ↓
[7. Evaluasi Training Set (MAPE, AIC, SIC)]
        ↓
[8. Robustness Testing (Out-of-Sample)]
        ↓
[9. Implementasi untuk Trading]
```

---

## Langkah 1 — Kumpulkan Data

### Variabel Dependen
| Variabel | Sumber | Periode |
|---|---|---|
| Bitcoin Low Price (Harga Terendah Bulanan) | Investing.com | Oktober 2012 – Oktober 2023 |

### Variabel Independen Utama (untuk ARDL)
| Variabel | Sumber |
|---|---|
| Bitcoin Open Price (Harga Pembuka) | Investing.com |

> **Catatan:** Dari 23 variabel independen yang diuji (endogen & eksogen), **Bitcoin Open Price** terpilih sebagai prediktor tunggal terkuat berdasarkan seleksi MAPE individual < 10%.

### Variabel Independen Lengkap (untuk referensi OLS)
**Endogen (internal Bitcoin network):**
Open Price, Stripped Block Size, Mining Difficulty, Hash Rate, Market Capitalization, Miners' Revenue, Number of Transactions, Circulating Supply, Total Transaction Fees, Price Volatility

**Eksogen (eksternal):**
U.S Federal Funds Rate, U.S GDP, U.S CPI Inflation Rate, U.S M2 Money Supply, Dow Jones Index, Gold Price & Volatility, Oil Price & Volatility, S&P500 Index, U.S 10-Year Bond Yield, USD Index, Google Trend

---

## Langkah 2 — Preprocessing Data

1. **Import data** ke software statistik (digunakan: SAS & EViews)
2. **Konversi frekuensi** — data harian dikonversi ke frekuensi bulanan
3. **Bersihkan data:**
   - Hapus missing values
   - Hapus duplikat
4. **Log transformasi** pada semua variabel:

```
Log(Bitcoin Low Price) → variabel dependen
Log(Bitcoin Open Price) → variabel independen
```

> Transformasi log dilakukan agar model lebih mudah mendeteksi tren dalam data dan menghasilkan prediksi lebih akurat.

5. **Dataset final ARDL Bull Monthly:**
   - Rentang data penuh: Oktober 2012 – Oktober 2023
   - 133 observasi bulanan
   - **Data training (Bull Run):** Maret 2020 – November 2021

---

## Langkah 3 — Seleksi Variabel Independen

Regresikan masing-masing dari 23 variabel independen secara individual terhadap Bitcoin Low Price, lalu hitung MAPE-nya:

```
Kriteria:
  MAPE < 10%  → variabel dipertimbangkan
  MAPE > 10%  → variabel dibuang

→ Variabel dengan MAPE terendah = Bitcoin Open Price
→ Hanya 1 variabel independen yang dimasukkan ke model ARDL
```

---

## Langkah 4 — Uji Stasioneritas (Augmented Dickey-Fuller Test)

Lakukan ADF Test pada **tingkat signifikansi 5%** untuk:
- Bitcoin Low Price
- Bitcoin Open Price

**Hasil yang diharapkan:**

```
Semua variabel → Stasioner pada First Difference
P-value = 0.00 (< 0.05) ✓
```

Jika variabel stasioner pada first difference, transformasikan data ke first difference sebelum melanjutkan ke langkah berikutnya.

---

## Langkah 5 — Analisis Correlogram (Penentuan Lag Optimal)

Buat **correlogram dari Partial Autocorrelation Function (PACF)** untuk:

1. **Bitcoin Low Price (first difference)** → menentukan lag dependen (p)
2. **Bitcoin Open Price (first difference)** → menentukan lag independen (q)

**Hasil lag optimal untuk prediksi Low Price bulanan:**

```
Model: General ARDL (5,2)
  → 5 lag untuk Bitcoin Low Price
  → 2 lag untuk Bitcoin Open Price
```

> Lag struktur (5,2) ini digunakan oleh ARDL Bull Monthly, General ARDL Monthly, maupun ARDL Bear Monthly — yang membedakan adalah **periode training**-nya.

---

## Langkah 6 — Pembangunan Model ARDL Bull Monthly

### Mengapa menggunakan Bull Market sebagai training set?

| Alasan | Penjelasan |
|---|---|
| Belajar tren terkini | Model dilatih pada bull run paling baru sehingga menangkap pola pasar terkini |
| Generalisasi | Struktur lag diambil dari data penuh (2012–2023), bukan hanya bull run |
| Uji lintas kondisi | Model tetap diuji pada kondisi bear dan sideway untuk membuktikan adaptabilitas |

### Spesifikasi Model

```
Variabel Dependen  : Log(BTC_Low_t)
Variabel Independen: Log(BTC_Open_t), Log(BTC_Open_t-1), Log(BTC_Open_t-2)
Lag Dependen       : Log(BTC_Low_t-1), ..., Log(BTC_Low_t-5)

Model              : ARDL(5,2)
Transformasi       : Log-Log
Software           : EViews
```

### Periode Training

```
Data Lag Struktur  : Oktober 2012 – Oktober 2023 (full dataset)
Data Training Model: Maret 2020 – November 2021 (Bull Run 2020)
```

---

## Langkah 7 — Evaluasi Training Set

Bandingkan performa semua model menggunakan tiga metrik:

| Metrik | Fungsi | Semakin kecil = |
|---|---|---|
| **MAPE** | Rata-rata persentase error absolut | Semakin akurat |
| **AIC** | Goodness-of-fit + penalti kompleksitas | Semakin baik |
| **SIC** | Mirip AIC, penalti lebih besar | Semakin baik |

### Hasil Perbandingan Model (Monthly Frequency — BTC Low Price)

| Model | MAPE | AIC | SIC |
|---|---|---|---|
| OLS | 0.3656% | -4.2137 | -3.5240 |
| General ARDL (5,2) | 0.5199% | -2.3754 | -2.1749 |
| **ARDL Bull (5,2)** ✓ | **0.1303%** | **-4.4165** | **-3.9688** |
| ARDL Bear (5,2) | 0.2608% | -3.7528 | -3.3246 |

**→ ARDL Bull Monthly terpilih sebagai model terbaik untuk frekuensi bulanan.**

---

## Langkah 8 — Robustness Testing (Out-of-Sample)

Model diuji pada **7 kondisi pasar berbeda** menggunakan data yang tidak pernah dilihat model sebelumnya (data Investing.com):

### Back Tests (Data Masa Lalu)

| Kondisi Pasar | Periode | MAPE |
|---|---|---|
| 2012 Bull Run | Agustus 2012 – Desember 2013 | 6.05% |
| 2013 Bear Run | Desember 2013 – Mei 2016 | 4.59% |
| 2016 Bull Run | Mei 2016 – Desember 2017 | 1.63% |
| 2017 Bear Run | Desember 2017 – Maret 2020 | 2.02% |
| Full Historical Life | April 2011 – Maret 2020 | 6.73% |

### Forward Tests (Data Masa Depan)

| Kondisi Pasar | Periode | MAPE |
|---|---|---|
| 2021 Bear Run | November 2021 – Desember 2022 | 1.47% |
| 2023 Mini Bull Run | Januari 2023 – Maret 2024 | 0.47% |

```
Rata-rata MAPE keseluruhan = 3.28%
Semua nilai MAPE < 10% (benchmark Lewis, 1982) ✓
```

---

## Langkah 9 — Implementasi untuk Trading

### Strategi Penggunaan Model

```
Prediksi model → Harga Low Bulanan (dalam log scale)
                       ↓
               Konversi balik (exp)
                       ↓
         Gunakan sebagai BUY LIMIT (harga beli)
                       ↓
         Tunggu harga turun menyentuh level prediksi
                       ↓
         Jual di harga penutupan bulan (BTC Close)
```

### Contoh Hasil Trading Demo

| Kondisi Pasar | Predicted Low | Actual Low | Actual Close | Profit (1 BTC) | Profit (100 BTC) |
|---|---|---|---|---|---|
| 2012 Bull Run | $13.40 | $13.20 | $20.40 | $7.00 | $700 |
| 2013 Bear Run | $221.01 | $221.00 | $264.10 | $43.09 | $4,309 |
| 2016 Bull Run | $744.15 | $741.10 | $963.40 | $219.25 | $21,925 |
| 2017 Bear Run | $3,725.33 | $3,681.80 | $4,102.30 | $376.97 | $37,697 |
| 2021 Bear Run | $18,396.44 | $18,207.90 | $20,496.30 | $2,099.86 | $209,986 |
| 2023 Mini Bull | $29,044.20 | $28,890.70 | $29,232.40 | $188.20 | $18,820 |

> *Biaya trading tidak dihitung dalam simulasi ini.*

---

## Keterbatasan Model

| Keterbatasan | Penjelasan | Solusi |
|---|---|---|
| **Overfitting ringan** | Performa training set lebih baik dari testing set | Dapat diterima karena out-of-sample tetap sangat baik |
| **Lag perlu diperbarui** | Karakteristik data dapat berubah seiring waktu | Perbarui lag secara berkala sesuai kondisi pasar terkini |
| **Perlu retraining** | Bull run terbaru mungkin berbeda dari 2020 | Latih ulang pada bull run paling baru saat tersedia |
| **Prediksi tidak 100% akurat** | Kadang overprediksi atau underprediksi | Kombinasikan dengan indikator teknikal untuk konfirmasi |

---

## Ringkasan Satu Halaman

```
TUJUAN      : Prediksi harga terendah BTC per bulan
MODEL       : ARDL Bull Monthly — ARDL(5,2)
VARIABEL    : Log(BTC Low) ~ Log(BTC Open) + lag-lagnya
TRAINING    : Maret 2020 – November 2021 (Bull Run 2020)
LAG STRUTUR : Dari data penuh Oktober 2012 – Oktober 2023
SOFTWARE    : EViews (model) + Excel (robustness test)
AKURASI     : Rata-rata MAPE 3.28% dari 7 uji out-of-sample
PENGGUNAAN  : Buy limit pada harga prediksi, exit di close bulanan
```

---

*Sumber: Rice, T. (2025). Optimal Prediction of Bitcoin High and Low Prices: An Exploratory Analysis. Dissertation DBA05/2025, Sacred Heart University.*
