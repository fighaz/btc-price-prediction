# Terminologi

Daftar istilah teknis yang digunakan dalam sistem prediksi harga Low BTC/IDR berbasis ARDL-ECM.

---

## Model & Statistik

### ARDL (Autoregressive Distributed Lag)
Model regresi time series yang menggabungkan dua komponen: lag dari variabel target itu sendiri (autoregressive) dan lag dari variabel-variabel penjelas (distributed lag). Dalam sistem ini, ARDL memodelkan harga Low bulanan sebagai fungsi dari Low bulan-bulan sebelumnya dan nilai historis Open, Close, serta Volume.

### ECM (Error Correction Model)
Representasi alternatif dari model ARDL yang memisahkan dinamika jangka pendek dari hubungan keseimbangan jangka panjang. ECM menjelaskan seberapa cepat harga Low kembali ke keseimbangannya setelah mengalami guncangan (shock). Digunakan setelah kointegrasi dikonfirmasi.

### UECM (Unrestricted Error Correction Model)
Bentuk ECM yang belum direstriksi, digunakan sebagai dasar untuk menjalankan bounds test kointegrasi. Semua variabel dalam level dan first-difference dimasukkan tanpa pembatasan koefisien.

### Kointegrasi
Kondisi di mana dua atau lebih variabel non-stasioner bergerak bersama dalam jangka panjang dan memiliki hubungan keseimbangan yang stabil. Meskipun masing-masing variabel bisa menyimpang dalam jangka pendek, kombinasi liniernya tetap stasioner. Dikonfirmasi melalui bounds test.

### Bounds Test (Pesaran-Shin-Smith)
Uji statistik untuk mendeteksi kointegrasi antara variabel endogen dan exogen dalam kerangka ARDL. Menggunakan F-statistic yang dibandingkan dengan dua batas kritis: batas bawah (semua variabel I(0)) dan batas atas (semua variabel I(1)). Tidak memerlukan semua variabel berada pada orde integrasi yang sama.

### Stasioneritas
Sifat suatu deret waktu di mana mean, variansi, dan autokovariansi tidak berubah seiring waktu. Model time series umumnya mensyaratkan variabel stasioner agar estimasi tidak palsu (spurious).

### ADF Test (Augmented Dickey-Fuller)
Uji hipotesis untuk menentukan apakah suatu deret waktu stasioner atau tidak. Hipotesis nol: deret mengandung unit root (tidak stasioner). Jika p-value < 0.05, deret dinyatakan stasioner.

### I(0), I(1), I(2)
Orde integrasi suatu variabel:
- **I(0)**: Stasioner di level — tidak perlu differencing
- **I(1)**: Stasioner setelah first-difference (dikurangi nilai bulan sebelumnya)
- **I(2)**: Stasioner setelah second-difference — jarang terjadi, umumnya mengindikasikan masalah data

### AIC (Akaike Information Criterion)
Kriteria pemilihan model yang menyeimbangkan kebaikan fit (log-likelihood) dengan kompleksitas model (jumlah parameter). Nilai AIC lebih kecil = model lebih baik. Digunakan dalam sistem ini untuk memilih lag optimal secara otomatis.

### BIC (Bayesian Information Criterion)
Alternatif AIC dengan penalti lebih besar terhadap jumlah parameter. Cenderung menghasilkan model dengan lag lebih sedikit dibanding AIC. Tersedia sebagai opsi di UI tetapi dinonaktifkan sementara.

### Lag
Nilai suatu variabel pada periode waktu sebelumnya. Lag-1 berarti nilai bulan lalu, lag-2 berarti dua bulan lalu, dst. Dalam ARDL, lag menangkap pengaruh historis terhadap nilai saat ini.

### Endog Lag (p)
Jumlah lag dari variabel endogen (Low) yang dimasukkan ke dalam model ARDL. Dipilih otomatis via AIC dengan batas maksimum 4.

### Exog Orders (q)
Jumlah lag dari masing-masing variabel exogen (Open, Close, Volume) yang dimasukkan ke model. Setiap variabel bisa memiliki jumlah lag berbeda. Contoh: `{'log_Open': 1, 'log_Close': 1, 'log_Volume': 0}`.

---

## Parameter Model

### λ (Lambda ECM / Speed of Adjustment)
Koefisien dari lagged level term dalam ECM, selalu bernilai negatif. Menunjukkan proporsi ketidakseimbangan jangka panjang yang dikoreksi dalam satu bulan. Contoh: λ = −0.15 berarti 15% deviasi dari keseimbangan terkoreksi setiap bulan.

### Half-Life
Waktu (dalam bulan) yang dibutuhkan agar deviasi dari keseimbangan jangka panjang berkurang sebesar 50%. Dihitung dari λ: `half-life = log(0.5) / log(1 + λ)`. Semakin kecil half-life, semakin cepat pasar kembali ke keseimbangan.

### Koefisien Jangka Panjang
Besarnya pengaruh permanen suatu variabel exogen terhadap Low dalam keseimbangan jangka panjang. Dihitung dari koefisien UECM: `lr_coef = −koefisien_exog / λ`. Dapat diinterpretasikan sebagai elastisitas jika variabel dalam log.

### R² (R-squared)
Proporsi variansi variabel target (Low) yang dapat dijelaskan oleh model. Nilai 0–1; semakin mendekati 1 semakin baik fit in-sample. Digunakan sebagai indikator kualitas fitting, bukan satu-satunya ukuran.

### Bias Correction (σ²/2)
Koreksi yang diterapkan saat mengkonversi prediksi dari skala log kembali ke skala Rupiah. Tanpa koreksi ini, rata-rata prediksi akan selalu lebih rendah dari nilai sebenarnya akibat Jensen's inequality. Formula: `Predicted_Low = exp(ŷ + σ²/2)`, di mana σ² adalah variansi residual model.

### Log-Transform
Transformasi variabel dengan mengambil logaritma natural sebelum dimasukkan ke model. Menstabilkan variansi yang cenderung membesar seiring kenaikan harga, membuat distribusi residual lebih mendekati normal, dan memungkinkan koefisien diinterpretasikan sebagai elastisitas.

### Rolling Window
Pilihan untuk membatasi data training hanya pada N tahun terakhir, bukan seluruh data sejak 2014. Berguna jika perilaku pasar dianggap telah berubah struktural. Nilai 0 berarti gunakan seluruh data.

---

## Variabel dalam Model

### Endogen (Y)
Variabel yang diprediksi oleh model. Dalam sistem ini: **Low bulanan** — harga terendah BTC/IDR dalam satu bulan kalender. Digunakan dalam bentuk log (`log_Low`) saat log-transform aktif.

### Exogen (X)
Variabel penjelas yang diasumsikan mempengaruhi Low tetapi tidak dipengaruhi balik oleh Low dalam model ini. Terdiri dari:
- **Open**: harga pembukaan hari pertama bulan
- **Close**: harga penutupan hari terakhir bulan
- **log_Volume**: log total volume transaksi bulanan

### Open Aktual (Injeksi)
Harga pembukaan BTC/IDR pada hari pertama bulan yang sedang diprediksi. Diambil langsung dari API pada saat model dijalankan dan diinjeksikan ke dalam exog_future menggantikan proyeksi VAR. Ini adalah sinyal nyata yang membedakan model ini dari forecast murni.

---

## Forecast & Evaluasi

### Confidence Interval (CI) 95%
Rentang nilai di mana harga Low aktual diperkirakan akan jatuh dengan probabilitas 95%, berdasarkan distribusi prediksi model. Dinyatakan sebagai `CI_Lower` dan `CI_Upper`. Semakin sempit interval, semakin presisi prediksi.

### CI Coverage
Persentase bulan (dalam backtest) di mana harga Low aktual jatuh di dalam CI 95%. Nilai ideal mendekati 95%. Coverage jauh di bawah 95% mengindikasikan CI terlalu sempit (model terlalu yakin).

### MAPE (Mean Absolute Percentage Error)
Rata-rata persentase selisih absolut antara prediksi dan aktual: `mean(|aktual − pred| / aktual × 100)`. Tidak bergantung skala harga sehingga mudah diinterpretasikan. Contoh: MAPE 8% berarti rata-rata prediksi meleset 8% dari harga aktual.

### RMSE (Root Mean Squared Error)
Akar dari rata-rata kuadrat selisih prediksi dan aktual, dalam satuan Rupiah. Memberi bobot lebih besar pada kesalahan besar dibanding MAE.

### MAE (Mean Absolute Error)
Rata-rata selisih absolut prediksi dan aktual dalam satuan Rupiah. Lebih robust terhadap outlier dibanding RMSE.

### Walk-Forward Backtest
Metode evaluasi model yang mensimulasikan penggunaan nyata: untuk setiap bulan dalam periode backtest, model dilatih ulang hanya dengan data yang tersedia sebelum bulan itu, lalu memprediksi bulan target. Hasilnya mencerminkan akurasi yang sesungguhnya akan didapat investor.

### Cointegration Rate
Persentase iterasi dalam backtest yang menghasilkan status kointegrasi (COINTEGRATED atau INCONCLUSIVE). Tingkat kointegrasi tinggi mengindikasikan hubungan jangka panjang yang konsisten antara Low dan variabel exogen.

---

## Proyeksi Exogen

### VAR (Vector Autoregression)
Model multivariate time series yang memproyeksikan beberapa variabel secara bersamaan, dengan setiap variabel dijelaskan oleh lag dari seluruh variabel dalam sistem. Digunakan dalam sistem ini untuk memproyeksikan Open, Close, dan Volume bulan depan sebelum diinjeksikan Open aktual.

### Exog Future
DataFrame 1 baris yang berisi nilai proyeksi variabel exogen untuk bulan target, digunakan sebagai input `exog_oos` dalam `fit.get_prediction()`. Kolom `log_Open` dalam DataFrame ini selalu diganti dengan nilai aktual sebelum forecast dijalankan.

---

## Simulasi DCA

### DCA (Dollar-Cost Averaging)
Strategi investasi di mana investor membeli aset dengan jumlah uang tetap secara berkala (dalam sistem ini: setiap bulan), tanpa memperhatikan harga saat itu. Tujuannya meratakan harga beli rata-rata sepanjang waktu dan mengurangi dampak volatilitas.

### Average Cost
Total Rupiah yang diinvestasikan dibagi total BTC yang dimiliki. Metrik utama efisiensi strategi DCA — semakin rendah average cost, semakin banyak BTC yang diperoleh per Rupiah.

### Hit (Strategi Model)
Kondisi di mana prediksi model berhasil: `predicted_low ≥ low_aktual`. Artinya harga aktual turun ke level prediksi atau lebih rendah dalam bulan tersebut, sehingga investor bisa membeli di harga prediksi.

### Miss (Strategi Model)
Kondisi di mana prediksi model terlalu optimis: `predicted_low < low_aktual`. Harga aktual tidak pernah turun serendah prediksi. Sistem melakukan fallback ke harga Close bulan itu agar total investasi tetap identik antar strategi (perbandingan fair).

### Hit Rate
Persentase bulan di mana strategi model berhasil (Hit) dari total bulan yang ada prediksi: `Hit / (Hit + Miss) × 100%`. Mengukur seberapa sering prediksi Low model dapat direalisasikan oleh investor.

### Return (%)
Keuntungan atau kerugian relatif investasi pada akhir periode: `(nilai portofolio − total investasi) / total investasi × 100`. Nilai portofolio dihitung sebagai total BTC × harga penutupan bulan terakhir.

### Modal Pending
Akumulasi modal yang belum diinvestasikan pada strategi model. Terjadi ketika tidak ada prediksi untuk bulan tersebut. Dengan logika saat ini modal langsung diinvestasikan setiap bulan (Hit atau fallback ke Close), sehingga Modal Pending selalu 0.

### Align to Model
Opsi dalam simulasi DCA yang memastikan semua strategi hanya membeli di bulan-bulan yang memiliki prediksi model. Memungkinkan perbandingan apel-ke-apel: semua strategi menanggung risiko dan peluang yang sama persis.

---

## Resample & Data

### OHLCV
Singkatan dari Open, High, Low, Close, Volume — komponen standar data harga aset keuangan per periode waktu:
- **Open**: harga saat periode dimulai
- **High**: harga tertinggi dalam periode
- **Low**: harga terendah dalam periode
- **Close**: harga saat periode berakhir
- **Volume**: total unit yang diperdagangkan

### Drop Partial
Penghapusan bulan yang datanya belum lengkap (< 20 hari trading). Diterapkan saat menyiapkan data training agar model tidak belajar dari bulan yang belum selesai. Dinonaktifkan (`drop_partial=False`) saat data bulan target dibutuhkan untuk evaluasi backtest.

### Forward-Fill
Teknik pengisian nilai yang hilang (missing value) dengan menggunakan nilai terakhir yang tersedia. Diterapkan pada hari libur dan gap dalam data harian serta pada bulan-bulan dengan volume nol.
