"""
Engine simulasi Dollar-Cost Averaging (DCA) BTC/IDR.

Membandingkan 4 strategi beli per bulan dengan modal yang sama:
    1. open  — beli di harga Open hari pertama bulan
    2. close — beli di harga Close hari terakhir bulan
    3. low   — beli di Low bulanan (timing sempurna, benchmark ideal)
    4. model — beli di predicted_low dari model ARDL-ECM (bila tersedia)

Tidak ada Streamlit di modul ini.
"""
import numpy as np
import pandas as pd

# Nama tampilan per strategi
STRATEGY_LABELS = {
    "open":  "Awal Bulan (Open)",
    "close": "Akhir Bulan (Close)",
    "low":   "Low Bulanan (Ideal)",
    "model": "Prediksi Model",
}

# Warna Altair per strategi
STRATEGY_COLORS = {
    "open":  "#2196F3",   # biru
    "close": "#78909C",   # abu
    "low":   "#4CAF50",   # hijau
    "model":  "#E91E63",   # merah
}


def simulate_dca(monthly_df, daily_df, modal_per_bulan,
                 pred_df=None, align_to_model=False):
    """
    Simulasi DCA untuk semua strategi.

    Parameters
    ----------
    monthly_df : DataFrame berindeks bulan (freq M), kolom Open/High/Low/Close.
    daily_df   : DataFrame dengan kolom Time (Timestamp), Open, High, Low, Close.
                 (dipertahankan untuk kompatibilitas, tidak dipakai lagi setelah
                 strategi random dihapus).
    modal_per_bulan : float — Rupiah diinvestasikan setiap bulan.
    pred_df    : DataFrame opsional dengan kolom Target_Month (date/str) +
                 Predicted_Low. Bila None, strategi model dilewati.
    align_to_model : bool — bila True, semua strategi hanya beli di bulan yang
                 ada prediksinya (perbandingan adil apel ke apel). Bila False,
                 strategi non-model beli setiap bulan tanpa kecuali.

    Returns
    -------
    dict dengan key = nama strategi ('open','close','low','model')
    dan value = DataFrame per-bulan:
        Bulan (str YYYY-MM), Harga_Beli, BTC_Dibeli, Total_BTC, Total_Invested,
        Average_Cost, Status, Modal_Pending

    Catatan: saat prediksi miss (pred < low aktual), strategi model fallback
    ke harga Close bulan itu agar total_invested identik antar strategi
    (perbandingan fair untuk keperluan akademis/skripsi).
    """
    # Siapkan lookup prediksi model (bulan YYYY-MM -> harga)
    pred_lookup = {}
    model_months = set()  # bulan yang ada prediksinya
    if pred_df is not None and not pred_df.empty:
        p = pred_df.copy()
        p["_key"] = pd.to_datetime(p["Target_Month"]).dt.strftime("%Y-%m")
        p_indexed = p.set_index("_key")["Predicted_Low"]
        pred_lookup = p_indexed.to_dict()
        model_months = set(pred_lookup.keys())

    strategies = {
        "open":  {},
        "close": {},
        "low":   {},
    }
    if pred_lookup:
        strategies["model"] = {}

    # Akumulator per strategi
    accum = {s: {"total_btc": 0.0, "total_invested": 0.0} for s in strategies}
    rows = {s: [] for s in strategies}

    # Tracker modal untuk strategi model; dengan fallback Close,
    # nilai ini selalu kembali ke 0 setiap bulan (tidak menumpuk).
    model_modal_pending = 0.0

    for ts, row in monthly_df.iterrows():
        bulan = ts.strftime("%Y-%m")
        y, m = ts.year, ts.month
        low_aktual = row["Low"]  # dipakai untuk cek apakah prediksi model tercapai

        # Mode align_to_model: strategi non-model hanya beli di bulan ada prediksinya
        if align_to_model and model_months and bulan not in model_months:
            continue

        prices = {
            "open":  row["Open"],
            "close": row["Close"],
            "low":   low_aktual,
        }

        for s in strategies:
            if s == "model":
                # Strategi model: cek apakah prediksi ada & tercapai
                pred_harga = pred_lookup.get(bulan, np.nan)
                model_modal_pending += modal_per_bulan  # tambah modal bulan ini

                if pd.isna(pred_harga) or pred_harga <= 0:
                    # Tidak ada prediksi bulan ini — skip, modal carry-over
                    rows[s].append({
                        "Bulan":           bulan,
                        "Harga_Prediksi":  np.nan,
                        "Harga_Beli":      np.nan,
                        "BTC_Dibeli":      0.0,
                        "Total_BTC":       accum[s]["total_btc"],
                        "Total_Invested":  accum[s]["total_invested"],
                        "Average_Cost":    accum[s]["total_invested"] / accum[s]["total_btc"]
                            if accum[s]["total_btc"] > 0 else np.nan,
                        "Status":          "Tidak ada prediksi",
                        "Modal_Pending":   model_modal_pending,
                    })
                    continue

                if pred_harga < low_aktual:
                    # MISS: prediksi terlalu rendah, harga tidak pernah serendah itu.
                    # Fallback ke Close agar total_invested identik dengan strategi lain.
                    harga_fallback = row["Close"]
                    btc = modal_per_bulan / harga_fallback
                    accum[s]["total_btc"] += btc
                    accum[s]["total_invested"] += modal_per_bulan
                    model_modal_pending -= modal_per_bulan  # tidak menumpuk
                    rows[s].append({
                        "Bulan":           bulan,
                        "Harga_Prediksi":  pred_harga,
                        "Harga_Beli":      harga_fallback,
                        "BTC_Dibeli":      btc,
                        "Total_BTC":       accum[s]["total_btc"],
                        "Total_Invested":  accum[s]["total_invested"],
                        "Average_Cost":    accum[s]["total_invested"] / accum[s]["total_btc"],
                        "Status":          "Miss → beli di Close",
                        "Modal_Pending":   model_modal_pending,
                    })
                else:
                    # HIT: prediksi >= low aktual — beli di harga prediksi
                    modal_beli = model_modal_pending
                    btc = modal_beli / pred_harga
                    accum[s]["total_btc"] += btc
                    accum[s]["total_invested"] += modal_beli
                    rows[s].append({
                        "Bulan":           bulan,
                        "Harga_Prediksi":  pred_harga,
                        "Harga_Beli":      pred_harga,
                        "BTC_Dibeli":      btc,
                        "Total_BTC":       accum[s]["total_btc"],
                        "Total_Invested":  accum[s]["total_invested"],
                        "Average_Cost":    accum[s]["total_invested"] / accum[s]["total_btc"],
                        "Status":          "Hit → beli di prediksi",
                        "Modal_Pending":   0.0,
                    })
                    model_modal_pending = 0.0  # reset setelah berhasil beli
            else:
                harga = prices.get(s, np.nan)
                if pd.isna(harga) or harga <= 0:
                    continue

                btc = modal_per_bulan / harga
                accum[s]["total_btc"] += btc
                accum[s]["total_invested"] += modal_per_bulan

                rows[s].append({
                    "Bulan":          bulan,
                    "Harga_Beli":     harga,
                    "BTC_Dibeli":     btc,
                    "Total_BTC":      accum[s]["total_btc"],
                    "Total_Invested": accum[s]["total_invested"],
                    "Average_Cost":   accum[s]["total_invested"] / accum[s]["total_btc"],
                    "Status":         "Beli",
                    "Modal_Pending":  0.0,
                })

    result = {}
    for s, data in rows.items():
        if data:
            df = pd.DataFrame(data)
            if "Status" not in df.columns:
                df["Status"] = "Beli"
            if "Modal_Pending" not in df.columns:
                df["Modal_Pending"] = 0.0
            if "Harga_Prediksi" not in df.columns:
                df["Harga_Prediksi"] = np.nan
            result[s] = df

    return result


def summary_dca(dca_result, harga_akhir):
    """
    Ringkasan metrik akhir periode untuk setiap strategi.

    Parameters
    ----------
    dca_result  : dict hasil dari simulate_dca()
    harga_akhir : float — harga Close bulan terakhir periode (untuk hitung ROI)

    Returns
    -------
    DataFrame dengan kolom:
        Strategi, Label, Total_Invested, Total_BTC, Average_Cost,
        Portfolio_Value, Return_Pct, Hit_Count, Miss_Count, Hit_Rate_Pct
    Diurutkan: Return_Pct descending (strategi terbaik di atas).
    """
    rows = []
    for s, df in dca_result.items():
        last = df.iloc[-1]
        total_invested = last["Total_Invested"]
        total_btc = last["Total_BTC"]
        avg_cost = last["Average_Cost"]
        portfolio_value = total_btc * harga_akhir
        return_pct = (portfolio_value - total_invested) / total_invested * 100

        # Hitung hit/miss hanya untuk strategi model
        hit_count = miss_count = None
        hit_rate_pct = None
        if s == "model" and "Status" in df.columns:
            hit_count  = int(df["Status"].str.startswith("Beli").sum())
            miss_count = int(df["Status"].str.startswith("Miss").sum())
            total_pred = hit_count + miss_count
            hit_rate_pct = hit_count / total_pred * 100 if total_pred > 0 else None

        rows.append({
            "Strategi":        s,
            "Label":           STRATEGY_LABELS.get(s, s),
            "Total_Invested":  total_invested,
            "Total_BTC":       total_btc,
            "Average_Cost":    avg_cost,
            "Portfolio_Value": portfolio_value,
            "Return_Pct":      return_pct,
            "Hit_Count":       hit_count,
            "Miss_Count":      miss_count,
            "Hit_Rate_Pct":    hit_rate_pct,
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("Return_Pct", ascending=False)
        .reset_index(drop=True)
    )


def build_long_df(dca_result):
    """
    Gabungkan hasil semua strategi menjadi satu DataFrame long format
    untuk dipakai chart Altair.

    Kolom: Bulan (str), Strategi (str), Label (str),
           Average_Cost, Total_BTC, Total_Invested, Harga_Beli

    Untuk strategi model, baris miss (Average_Cost NaN) di-forward-fill
    supaya garis chart tidak putus — average cost tetap flat sampai
    berhasil beli di bulan berikutnya.
    """
    frames = []
    for s, df in dca_result.items():
        d = df.copy()
        if s == "model":
            # Forward-fill kolom akumulatif agar garis tidak putus saat miss
            for col in ["Average_Cost", "Total_BTC", "Total_Invested"]:
                if col in d.columns:
                    d[col] = d[col].ffill()
        d["Strategi"] = s
        d["Label"] = STRATEGY_LABELS.get(s, s)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def project_next_month(dca_result, predicted_low, modal_per_bulan,
                       target_month_str):
    """
    Proyeksikan average cost bulan depan untuk tiap strategi bila beli
    di predicted_low (dari model ARDL-ECM).

    Returns DataFrame satu baris per strategi dengan kolom:
        Label, Current_AvgCost, New_AvgCost, Delta_Pct
    """
    rows = []
    for s, df in dca_result.items():
        last = df.iloc[-1]
        cur_total_btc = last["Total_BTC"]
        cur_total_inv = last["Total_Invested"]

        new_btc = modal_per_bulan / predicted_low
        new_total_btc = cur_total_btc + new_btc
        new_total_inv = cur_total_inv + modal_per_bulan
        new_avg = new_total_inv / new_total_btc

        rows.append({
            "Strategi":         s,
            "Label":            STRATEGY_LABELS.get(s, s),
            "Avg_Cost_Saat_Ini": last["Average_Cost"],
            "Avg_Cost_Baru":    new_avg,
            "Delta_Pct":        (new_avg - last["Average_Cost"]) / last["Average_Cost"] * 100,
        })
    return pd.DataFrame(rows)
