# 📊 Panduan Pendekatan ARDL yang Benar untuk Prediksi Harga Low Bitcoin

## 🎯 Tujuan

Membangun model yang **valid secara ekonometrika** untuk:

- Memprediksi **harga low harian**
- Mengestimasi **harga terendah bulanan (monthly low)**

---

# 🧠 1. Posisi Model Kamu Saat Ini

## ✅ Yang sudah benar

- Menggunakan data **OHLCV**
- Feature engineering (EMA, volatility, lag)
- ADF test (stationarity)
- Time series awareness

## ❌ Yang masih salah

- Menggunakan **Ridge (ML)** → bukan ARDL
- Tidak ada **cointegration test**
- Tidak ada **Error Correction Model (ECM)**
- Forecast pakai **recursive + proxy (tidak ekonometrika)**

---

# 🎯 2. Apa itu ARDL yang Benar?

Model dasar ARDL:

```
Y_t = α + Σ(φ_i * Y_{t-i}) + Σ(β_j * X_{t-j}) + ε_t
```

Contoh kasus kamu:

- Y = Low
- X = Open, Close, Volume

---

# 🔥 3. Perbedaan ARDL vs Model Kamu

| Aspek         | Model Kamu | ARDL yang Benar |
| ------------- | ---------- | --------------- |
| Estimasi      | Ridge (ML) | OLS             |
| Struktur      | Black-box  | Interpretatif   |
| Cointegration | ❌         | ✔️              |
| ECM           | ❌         | ✔️              |
| Forecast      | Recursive  | Dynamic         |

---

# 🚀 4. Langkah Metodologi yang Benar

---

## STEP 1 — Definisi Variabel

### Dependen:

- `Low_t`

### Independen:

- `Open_t`
- `Close_t`
- `Volume_t`
- (opsional: volatility)

---

## STEP 2 — Uji Stasioneritas

Gunakan:

- ADF Test

Syarat:

- Variabel harus I(0) atau I(1)
- ❌ Tidak boleh I(2)

---

## STEP 3 — Pemilihan Lag Optimal

Gunakan:

- AIC / BIC

Contoh:

```
ARDL(2,2,1)
```

---

## STEP 4 — Estimasi Model ARDL (OLS)

Contoh:

```
Low_t = α
      + β1 Low_{t-1} + β2 Low_{t-2}
      + γ1 Open_{t-1} + γ2 Open_{t-2}
      + δ1 Close_{t-1}
      + ε_t
```

Gunakan:

- statsmodels (Python)
- atau EViews

---

## STEP 5 — Bounds Test (WAJIB)

Tujuan:

- Menguji **cointegration (hubungan jangka panjang)**

Keputusan:

- F-stat > Upper bound → ✔️ cointegration
- F-stat < Lower bound → ❌ tidak ada hubungan

---

## STEP 6 — Error Correction Model (ECM)

Jika cointegration ada:

```
ΔLow_t = λ(ECM_{t-1})
       + Σ ΔLow
       + Σ ΔOpen
       + ε_t
```

Makna:

- λ = kecepatan kembali ke equilibrium
- Biasanya negatif

---

## STEP 7 — Forecasting

Gunakan:

- Dynamic forecasting dari ARDL / ECM

❌ Hindari:

- Recursive ML style
- Proxy OHLC buatan

---

# 🎯 5. Cara Mendapatkan Monthly Low

---

## Cara dasar:

```
Monthly Low = min(Low_t+1 ... Low_t+30)
```

---

## Upgrade (lebih kuat):

- Bootstrap residual
- Simulasi beberapa skenario

Output:

- Expected monthly low
- Worst-case low

---

# 🔧 6. Refactor Kode Kamu

---

## ❌ Yang harus dihapus:

- Ridge regression
- Proxy OHLC (synthetic data)
- Recursive forecasting

---

## ✅ Yang dipertahankan:

- Data pipeline
- Feature lag
- ADF test

---

## ✅ Yang ditambahkan:

- ARDL (OLS)
- Bounds test
- ECM

---

# 🧠 7. Justifikasi Akademik (untuk skripsi)

Gunakan narasi ini:

> Penelitian ini menggunakan model ARDL untuk menangkap hubungan jangka pendek dan jangka panjang antara harga terendah Bitcoin dengan variabel internal pasar (OHLCV). Model kemudian ditransformasikan ke dalam bentuk Error Correction Model (ECM) untuk menganalisis dinamika penyesuaian jangka pendek. Forecast harga harian digunakan untuk mengestimasi harga terendah bulanan.

---

# ⚠️ 8. Kesalahan yang Harus Dihindari

❌ Mengklaim ARDL tapi pakai ML
❌ Tidak ada cointegration test
❌ Forecast tanpa ECM
❌ Terlalu banyak variabel (overfitting)

---

# 🔚 9. Kesimpulan

## ✔️ Model yang benar:

- ARDL (OLS-based)
- Ada cointegration
- Ada ECM

## ✔️ Workflow:

1. Daily forecast (ARDL)
2. Ambil minimum → monthly low

## ❗ Fokus utama:

> Bukan menambah variabel, tapi memperbaiki metodologi

---

# 🚀 10. Next Step (Rekomendasi)

- Implementasi ARDL di Python (statsmodels)
- Tambahkan ECM
- Validasi dengan MAPE / RMSE
- (Opsional) Monte Carlo simulation

---

**Status kamu sekarang:**

> Sudah 70% benar secara engineering
> Tinggal upgrade ke 100% ekonometrika ✅
