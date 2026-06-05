"""
Narasi metodologi + builder tabel + export laporan pipeline (Markdown & PDF).

Modul ini menjadi sumber-tunggal untuk:
  - Teks narasi tiap tahap pipeline (dipakai di Prediksi Baru & Riwayat).
  - Fungsi pembangun DataFrame dari dict hasil (adf_results, ecm_info, dll.)
    supaya UI dan export laporan memakai tabel yang sama.
  - build_pipeline_report_md / build_pipeline_report_pdf untuk tombol download.

Tujuan thesis-grade: tiap tahap memuat informasi yang cukup untuk menjelaskan
MENGAPA hasilnya seperti itu, bukan hanya angka akhir.
"""
import io
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# NARASI TIAP TAHAP (dipakai bersama Prediksi Baru & Riwayat)
# ============================================================================
NARASI = {
    1: (
        "Data harga BTC/IDR diambil dari **Indodax API** dalam candle per-jam, "
        "lalu diagregasi menjadi harian (Open=harga pertama, High=maks, Low=min, "
        "Close=harga terakhir, Volume=jumlah). Nilai nol/kosong diganti NaN, "
        "diisi maju (forward-fill), lalu diinterpolasi linier agar deret harian rapat."
    ),
    2: (
        "Data harian diagregasi ke **bulanan**: Open = hari pertama bulan, "
        "High = maksimum bulan, **Low = minimum bulan (variabel target)**, "
        "Close = hari terakhir, Volume = total bulan, log_Volume = log1p(Volume). "
        "Bulan dengan data < 20 hari dibuang (drop partial). Rolling window "
        "membatasi data ke N tahun terakhir bila diaktifkan (0 = semua data)."
    ),
    3: (
        "Saat log-transform aktif, variabel diubah ke skala log: "
        "endogen = log(Low), eksogen = [log_Open, log_Close, log_Volume]. "
        "Transformasi log menstabilkan varians dan membuat hubungan lebih linier. "
        "Hasil prediksi di-inverse kembali ke Rupiah dengan koreksi bias "
        "exp(ŷ + σ²/2)."
    ),
    4: (
        "**Augmented Dickey-Fuller (ADF)** menguji keberadaan unit root "
        "(H0: deret tidak stasioner). Deret disebut stasioner bila statistik ADF "
        "lebih kecil dari nilai kritis (atau p-value < 0.05). Bila tidak stasioner "
        "di level, uji diulang pada selisih pertama untuk menentukan ordo integrasi "
        "I(d). ARDL bounds test sah selama tidak ada variabel I(2)."
    ),
    5: (
        "Lag endogen (p) dan order tiap eksogen (q) dipilih **bersamaan** lewat "
        "grid search `ardl_select_order`: seluruh kombinasi dievaluasi, lalu "
        "kombinasi dengan **information criterion (AIC/BIC) terkecil** dipilih "
        "(baris ✓). Model ARDL kemudian diestimasi via OLS; koefisien, t-stat, dan "
        "p-value menunjukkan signifikansi tiap suku."
    ),
    6: (
        "**Bounds test Pesaran-Shin-Smith** menguji adanya hubungan jangka panjang "
        "(kointegrasi). Statistik F dibandingkan dengan pita nilai kritis bawah/atas: "
        "F > batas atas → **kointegrasi**; F < batas bawah → **tidak kointegrasi**; "
        "di antara keduanya → **inconclusive**."
    ),
    7: (
        "Bila kointegrasi terpenuhi, model **Error Correction (ECM)** mengukur "
        "kecepatan penyesuaian λ menuju ekuilibrium (semakin negatif = semakin cepat) "
        "dan half-life (bulan untuk memulihkan 50% guncangan). Koefisien jangka panjang "
        "menunjukkan elastisitas. Diagnostik residual menguji autokorelasi "
        "(Breusch-Godfrey), heteroskedastisitas (ARCH), normalitas (Jarque-Bera), "
        "dan stabilitas struktural (CUSUM)."
    ),
    8: (
        "Variabel eksogen masa depan diproyeksikan dengan **VAR(p)** (lag dipilih via "
        "AIC). Khusus Open bulan target, **harga pembukaan aktual hari pertama** "
        "diinjeksikan menggantikan proyeksi VAR — karena di awal bulan nilai ini sudah "
        "diketahui, sehingga meningkatkan akurasi forecast."
    ),
    9: (
        "Forecast 1-langkah dihitung dari model ARDL menggunakan eksogen masa depan. "
        "Nilai pada skala log di-inverse ke Rupiah dengan koreksi bias exp(ŷ + σ²/2). "
        "Interval keyakinan 95% (α=0.05) memberi batas bawah & atas prediksi."
    ),
}


# ============================================================================
# BUILDER TABEL DARI DICT HASIL
# ============================================================================
def adf_to_df(adf_results):
    """adf_results dict -> DataFrame tabel ADF (level + diff + nilai kritis)."""
    if not adf_results:
        return pd.DataFrame()
    rows = []
    for name, r in adf_results.items():
        if not isinstance(r, dict) or "adf_stat" not in r:
            continue  # lewati key meta seperti "_any_i2"
        order = r.get("order")
        rows.append({
            "Variabel": name,
            "ADF stat": round(r["adf_stat"], 4),
            "Kritis 5%": round(r.get("crit_5"), 4) if r.get("crit_5") is not None else None,
            "p-value": round(r["p_value"], 4),
            "Stasioner (level)": "Ya" if r.get("level_stationary") else "Tidak",
            "ADF diff": round(r["diff_adf_stat"], 4) if "diff_adf_stat" in r else None,
            "p-value diff": round(r["diff_p_value"], 4) if "diff_p_value" in r else None,
            "Order I(d)": order,
            "n obs": r.get("nobs"),
        })
    return pd.DataFrame(rows)


def longrun_to_df(ecm_info):
    """ecm_info['long_run'] -> DataFrame koefisien jangka panjang."""
    if not ecm_info or not ecm_info.get("long_run"):
        return pd.DataFrame()
    rows = []
    for var, c in ecm_info["long_run"].items():
        rows.append({
            "Variabel": var,
            "UECM coef": round(c["uecm_coef"], 6),
            "LR coef": round(c["lr_coef"], 6),
            "t-stat": round(c["t"], 4),
            "p-value": round(c["p"], 4),
        })
    return pd.DataFrame(rows)


def residual_diag_to_df(ecm_info):
    """ecm_info -> DataFrame ringkas diagnostik residual (BG/ARCH/JB/CUSUM)."""
    if not ecm_info:
        return pd.DataFrame()
    rows = [
        {
            "Uji": "Breusch-Godfrey (autokorelasi)",
            "Statistik": round(ecm_info["bg_lm"], 4) if ecm_info.get("bg_lm") is not None else None,
            "p-value": round(ecm_info["bg_p"], 4) if ecm_info.get("bg_p") is not None else None,
        },
        {
            "Uji": "ARCH LM (heteroskedastisitas)",
            "Statistik": round(ecm_info["arch_lm"], 4) if ecm_info.get("arch_lm") is not None else None,
            "p-value": round(ecm_info["arch_p"], 4) if ecm_info.get("arch_p") is not None else None,
        },
        {
            "Uji": "Jarque-Bera (normalitas)",
            "Statistik": round(ecm_info["jb_stat"], 4) if ecm_info.get("jb_stat") is not None else None,
            "p-value": round(ecm_info["jb_p"], 4) if ecm_info.get("jb_p") is not None else None,
        },
        {
            "Uji": "CUSUM (stabilitas struktural)",
            "Statistik": "Stabil" if ecm_info.get("cusum_stable") else "Tidak stabil",
            "p-value": None,
        },
    ]
    return pd.DataFrame(rows)


def bounds_table_to_df(ardl_info):
    """ardl_info['bounds_crit_table'] -> DataFrame batas kritis untuk ditampilkan."""
    crit = ardl_info.get("bounds_crit_table")
    if crit is None or not isinstance(crit, pd.DataFrame):
        return pd.DataFrame()
    df = crit.copy()
    df = df.reset_index()
    df.columns = ["Persentil", "Batas bawah I(0)", "Batas atas I(1)"]
    df["Batas bawah I(0)"] = df["Batas bawah I(0)"].round(4)
    df["Batas atas I(1)"] = df["Batas atas I(1)"].round(4)
    return df


# ============================================================================
# STRUKTUR LAPORAN (sumber bersama untuk MD & PDF)
# ============================================================================
def _fmt_rp(x):
    return f"Rp {x:,.0f}" if x is not None and pd.notnull(x) else "N/A"


def build_report_sections(result, run_config):
    """Rakit list seksi laporan: [{judul, narasi, df (opsional), teks (opsional)}].

    Dipakai oleh build_pipeline_report_md dan build_pipeline_report_pdf agar
    konten identik antar format.
    """
    sections = []
    mode = run_config.get("mode", "forecast")

    if mode == "backtest":
        # Laporan backtest: metadata + tabel hasil + metrik.
        results_df = result.get("results_df")
        metrics = result.get("metrics")
        if metrics:
            mtxt = (
                f"MAPE={metrics['mape']:.2f}% · RMSE={_fmt_rp(metrics['rmse'])} · "
                f"MAE={_fmt_rp(metrics['mae'])} · CI coverage={metrics['ci_coverage']:.1f}% · "
                f"R²={metrics['r2']:.4f} · n={metrics['n']} bulan · "
                f"cointegrated={metrics['coint_rate']:.0f}% iterasi"
            )
        else:
            mtxt = "Tidak ada metrik (semua iterasi gagal)."
        sections.append({
            "judul": "Backtest Walk-Forward",
            "narasi": (
                "Backtest mengulang seluruh tahap pipeline (data → stasioneritas → "
                "pemilihan lag → estimasi → kointegrasi → forecast) untuk tiap bulan, "
                "memakai data s/d bulan target dan menginjeksikan Open aktual. "
                "Hasil prediksi dibandingkan dengan Low aktual."
            ),
            "teks": f"**Metrik agregat:** {mtxt}",
        })
        if results_df is not None and not results_df.empty:
            bt = results_df.copy()
            bt["month"] = pd.to_datetime(bt["month"]).dt.strftime("%Y-%m")
            sections.append({
                "judul": "Tabel Hasil per Bulan",
                "narasi": "",
                "df": bt,
            })
        return sections

    # ---- Mode forecast / forecast_and_evaluate ----
    ardl = result.get("ardl_info", {})
    daily = result.get("daily")
    monthly_full = result.get("monthly_full")
    monthly = result.get("monthly_history")

    # Tahap 1
    if daily is not None and not daily.empty:
        t1 = (
            f"Rentang: {daily.index.min().date()} s/d {daily.index.max().date()} · "
            f"Jumlah hari: {len(daily)}"
        )
    else:
        t1 = "Data harian tidak tersedia."
    sections.append({"judul": "Tahap 1 — Pengambilan & Kualitas Data Harian",
                     "narasi": NARASI[1], "teks": t1})

    # Tahap 2
    n_full = len(monthly_full) if monthly_full is not None else 0
    n_used = len(monthly) if monthly is not None else 0
    sections.append({
        "judul": "Tahap 2 — Resampling ke Bulanan & Rolling Window",
        "narasi": NARASI[2],
        "teks": f"Baris bulanan tersedia: {n_full} · dipakai untuk training: {n_used}",
        "df": monthly.tail(12).reset_index() if monthly is not None else None,
    })

    # Tahap 3
    sections.append({
        "judul": "Tahap 3 — Transformasi Variabel (log)",
        "narasi": NARASI[3],
        "teks": f"Log-transform: {'Aktif' if run_config.get('log_transform') else 'Tidak'}",
    })

    # Tahap 4
    sections.append({
        "judul": "Tahap 4 — Uji Stasioneritas (ADF)",
        "narasi": NARASI[4],
        "df": adf_to_df(result.get("adf_results")),
        "teks": ("⚠️ Terdeteksi variabel I(2) — ARDL bounds test menjadi tidak valid."
                 if (result.get("adf_results") or {}).get("_any_i2") else ""),
    })

    # Tahap 5
    t5 = (
        f"Lag endogen terpilih (p): **{ardl.get('endog_lag')}** · "
        f"Order eksogen: {ardl.get('exog_orders')} · "
        f"R² training: {ardl.get('train_r2'):.4f}"
        if ardl.get("train_r2") is not None else ""
    )
    sec5 = {"judul": "Tahap 5 — Pemilihan Lag (Endogen & Eksogen) & Estimasi ARDL",
            "narasi": NARASI[5], "teks": t5}
    sections.append(sec5)
    lag_ic = result.get("lag_ic_table")
    if lag_ic is not None and not lag_ic.empty:
        sections.append({"judul": "Tahap 5a — Tabel Kandidat Lag (Information Criterion)",
                         "narasi": f"Total {len(lag_ic)} kombinasi dievaluasi; baris ✓ = IC terkecil.",
                         "df": lag_ic})
    coef = result.get("coef_table")
    if coef is not None and not coef.empty:
        coef_disp = coef.copy()
        for c in ["coef", "t", "p"]:
            if c in coef_disp:
                coef_disp[c] = coef_disp[c].round(6)
        sections.append({"judul": "Tahap 5b — Koefisien Model ARDL",
                         "narasi": "", "df": coef_disp})

    # Tahap 6
    bounds_txt = (
        f"F-statistik: **{ardl.get('bounds_f_stat'):.4f}** · Verdict: "
        f"**{ardl.get('cointegration')}**"
        if ardl.get("bounds_f_stat") is not None else ""
    )
    sections.append({
        "judul": "Tahap 6 — Uji Kointegrasi (Bounds Test)",
        "narasi": NARASI[6],
        "teks": bounds_txt,
        "df": bounds_table_to_df(ardl),
    })

    # Tahap 7
    ecm = result.get("ecm_info")
    if ecm:
        hl = ecm.get("half_life_months")
        t7 = (
            f"λ (speed of adjustment): **{ecm['lambda']:.6f}** "
            f"(t={ecm['t']:.3f}, p={ecm['p']:.4f}) · "
            f"Half-life: {hl:.2f} bulan" if hl else
            f"λ (speed of adjustment): **{ecm['lambda']:.6f}** "
            f"(t={ecm['t']:.3f}, p={ecm['p']:.4f})"
        )
        sections.append({"judul": "Tahap 7 — Diagnostik ECM",
                         "narasi": NARASI[7], "teks": t7,
                         "df": longrun_to_df(ecm)})
        sections.append({"judul": "Tahap 7a — Diagnostik Residual",
                         "narasi": "", "df": residual_diag_to_df(ecm)})
    else:
        sections.append({"judul": "Tahap 7 — Diagnostik ECM",
                         "narasi": NARASI[7],
                         "teks": "Tidak kointegrasi → ECM tidak diinterpretasikan."})

    # Tahap 8
    vi = result.get("var_info") or {}
    exog_future = result.get("exog_future")
    t8 = (
        f"Lag VAR terpilih (k_ar): **{vi.get('k_ar')}**"
        + (" (fallback p=1)" if vi.get("fallback_p1") else "")
    )
    sec8 = {"judul": "Tahap 8 — Proyeksi Variabel Eksogen (VAR)",
            "narasi": NARASI[8], "teks": t8}
    if exog_future is not None and not exog_future.empty:
        sec8["df"] = exog_future.reset_index().rename(columns={"index": "Bulan"})
    sections.append(sec8)

    # Tahap 9
    fc = result.get("forecast")
    fd = result.get("forecast_detail") or {}
    if fc is not None and not fc.empty:
        row = fc.iloc[0]
        t9 = (
            f"ŷ (skala log): {fd.get('yhat_log')} · σ² (bias correction): {fd.get('sigma2')}\n\n"
            f"**Prediksi Low: {_fmt_rp(row['Predicted_Low'])}** · "
            f"CI 95%: [{_fmt_rp(row['CI_Lower'])} , {_fmt_rp(row['CI_Upper'])}]"
        )
    else:
        t9 = ""
    sec9 = {"judul": "Tahap 9 — Forecasting Akhir + Interval Keyakinan",
            "narasi": NARASI[9], "teks": t9}
    sections.append(sec9)

    # Evaluasi (bila ada)
    ev = result.get("eval_result")
    if ev:
        sections.append({
            "judul": "Evaluasi vs Aktual",
            "narasi": "",
            "teks": (
                f"Actual Low: {_fmt_rp(ev['actual_low'])} · "
                f"Error: {ev['error_pct']:.2f}% · "
                f"Dalam CI 95%: {'Ya' if ev['in_ci'] else 'Tidak'}"
            ),
        })

    return sections


# ============================================================================
# EXPORT MARKDOWN
# ============================================================================
def build_pipeline_report_md(result, run_config, run_id=None, run_at=None):
    """Rakit laporan pipeline lengkap sebagai string Markdown."""
    lines = ["# Laporan Rincian Pipeline ARDL-ECM", ""]
    meta = [
        f"- **Mode:** {run_config.get('mode')}",
        f"- **Train end:** {run_config.get('train_end_date')}",
        f"- **Log transform:** {run_config.get('log_transform')}",
        f"- **Rolling window (thn):** {run_config.get('rolling_window_years')}",
        f"- **Max lag endog/exog:** {run_config.get('max_lag_endog')}/{run_config.get('max_lag_exog')}",
        f"- **IC:** {run_config.get('ic')}",
    ]
    if run_id is not None:
        meta.insert(0, f"- **Run ID:** {run_id}")
    if run_at is not None:
        meta.insert(0, f"- **Waktu run:** {run_at}")
    lines += meta + [""]

    for sec in build_report_sections(result, run_config):
        lines.append(f"## {sec['judul']}")
        if sec.get("narasi"):
            lines += ["", sec["narasi"]]
        if sec.get("teks"):
            lines += ["", sec["teks"]]
        df = sec.get("df")
        if df is not None and not df.empty:
            try:
                lines += ["", df.to_markdown(index=False)]
            except Exception as e:
                logger.warning(f"Gagal to_markdown untuk {sec['judul']}: {e}")
                lines += ["", "```", df.to_string(index=False), "```"]
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# EXPORT PDF (reportlab platypus)
# ============================================================================
def build_pipeline_report_pdf(result, run_config, run_id=None, run_at=None):
    """Rakit laporan pipeline sebagai bytes PDF via reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=7.5, leading=10)
    flow = []

    flow.append(Paragraph("Laporan Rincian Pipeline ARDL-ECM", styles["Title"]))
    flow.append(Spacer(1, 4 * mm))

    meta = []
    if run_at is not None:
        meta.append(f"Waktu run: {run_at}")
    if run_id is not None:
        meta.append(f"Run ID: {run_id}")
    meta += [
        f"Mode: {run_config.get('mode')}",
        f"Train end: {run_config.get('train_end_date')}",
        f"Log transform: {run_config.get('log_transform')} · "
        f"IC: {run_config.get('ic')} · "
        f"Max lag endog/exog: {run_config.get('max_lag_endog')}/{run_config.get('max_lag_exog')}",
    ]
    for m in meta:
        flow.append(Paragraph(m, body))
    flow.append(Spacer(1, 4 * mm))

    def _df_to_table(df):
        data = [list(df.columns)]
        for _, r in df.iterrows():
            data.append(["" if pd.isnull(v) else str(v) for v in r.tolist()])
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2196F3")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return tbl

    for sec in build_report_sections(result, run_config):
        flow.append(Paragraph(sec["judul"], styles["Heading2"]))
        if sec.get("narasi"):
            flow.append(Paragraph(sec["narasi"], body))
        if sec.get("teks"):
            flow.append(Paragraph(sec["teks"].replace("\n", "<br/>"), body))
        df = sec.get("df")
        if df is not None and not df.empty:
            flow.append(Spacer(1, 2 * mm))
            try:
                flow.append(_df_to_table(df))
            except Exception as e:
                logger.warning(f"Gagal membuat tabel PDF {sec['judul']}: {e}")
                flow.append(Paragraph(df.to_string(index=False), small))
        flow.append(Spacer(1, 4 * mm))

    doc.build(flow)
    return buf.getvalue()
