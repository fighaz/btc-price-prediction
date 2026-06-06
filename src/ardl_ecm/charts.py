"""
Chart Altair interaktif untuk Streamlit (tooltip, zoom, pan).

Semua fungsi mengembalikan objek alt.Chart yang langsung bisa dipakai
st.altair_chart(chart, use_container_width=True).

Harga di-skala ke Miliar IDR (nilai / 1e9) agar sumbu mudah dibaca.
"""
import altair as alt
import pandas as pd

# Palet warna konsisten lintas halaman
HIST = "#2196F3"     # historis
PRED = "#E91E63"     # prediksi
ACTUAL = "#4CAF50"   # aktual
CI = "#E91E63"       # pita confidence interval
POS = "#4CAF50"      # error positif
NEG = "#F44336"      # error negatif

_SCALE = 1e9  # IDR -> Miliar IDR
_Y_TITLE = "Harga Low (Miliar IDR)"


def _month_label(series):
    """Konversi kolom tanggal -> string 'YYYY-MM'."""
    return pd.to_datetime(series).dt.strftime("%Y-%m")


def _month_series(df):
    """
    Ekstrak Series bulan target dari forecast DataFrame secara robust.

    Mendukung dua bentuk input:
      - DataFrame berindeks bulan (index = Timestamp bulan target)
      - DataFrame dengan kolom 'Target_Month'
    """
    if "Target_Month" in df.columns:
        return pd.to_datetime(df["Target_Month"])
    return pd.to_datetime(df.index.to_series())


def forecast_with_history_chart(monthly_history, forecast_df, eval_df=None,
                                history_months=24):
    """
    Chart utama: garis low historis bulanan + titik forecast + pita CI 95%
    + titik aktual (opsional).

    Parameters:
        monthly_history: DataFrame berindeks bulan dengan kolom 'Low'.
        forecast_df: DataFrame dengan kolom Predicted_Low, CI_Lower, CI_Upper —
                     berindeks bulan target ATAU punya kolom 'Target_Month'.
        eval_df: DataFrame opsional dengan kolom Target_Month, Actual_Low.
        history_months: jumlah bulan history terakhir yang ditampilkan.

    Returns alt.Chart interaktif.
    """
    hist = monthly_history.tail(history_months)
    hist_df = pd.DataFrame(
        {
            "Bulan": _month_label(hist.index.to_series()),
            "Low": hist["Low"].values / _SCALE,
            "Tipe": "Historis",
        }
    )

    fc_df = pd.DataFrame(
        {
            "Bulan": _month_label(_month_series(forecast_df)).values,
            "Low": forecast_df["Predicted_Low"].values / _SCALE,
            "CI_Lower": forecast_df["CI_Lower"].values / _SCALE,
            "CI_Upper": forecast_df["CI_Upper"].values / _SCALE,
            "Tipe": "Prediksi",
        }
    )

    base_x = alt.X("Bulan:O", title="Bulan", axis=alt.Axis(labelAngle=-45))
    layers = []

    # Pita CI prediksi
    if fc_df["CI_Lower"].notna().all() and fc_df["CI_Upper"].notna().all():
        layers.append(
            alt.Chart(fc_df)
            .mark_area(opacity=0.18, color=CI)
            .encode(
                x=base_x,
                y=alt.Y("CI_Lower:Q", title=_Y_TITLE),
                y2="CI_Upper:Q",
            )
        )

    # Garis + titik history
    hist_line = (
        alt.Chart(hist_df)
        .mark_line(point=alt.OverlayMarkDef(size=55, filled=True), color=HIST,
                   strokeWidth=2.5)
        .encode(
            x=base_x,
            y=alt.Y("Low:Q", title=_Y_TITLE),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Low:Q", title="Low (M IDR)", format=",.3f"),
                alt.Tooltip("Tipe:N", title="Tipe"),
            ],
        )
    )
    layers.append(hist_line)

    # Titik forecast (bintang besar)
    fc_point = (
        alt.Chart(fc_df)
        .mark_point(shape="diamond", size=320, filled=True, color=PRED)
        .encode(
            x=base_x,
            y="Low:Q",
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan target"),
                alt.Tooltip("Low:Q", title="Prediksi (M IDR)", format=",.3f"),
                alt.Tooltip("CI_Lower:Q", title="CI bawah", format=",.3f"),
                alt.Tooltip("CI_Upper:Q", title="CI atas", format=",.3f"),
            ],
        )
    )
    layers.append(fc_point)

    # Titik aktual (opsional)
    if eval_df is not None and not eval_df.empty:
        ev = pd.DataFrame(
            {
                "Bulan": _month_label(eval_df["Target_Month"]),
                "Low": eval_df["Actual_Low"] / _SCALE,
            }
        )
        actual_point = (
            alt.Chart(ev)
            .mark_point(shape="circle", size=170, filled=True, color=ACTUAL)
            .encode(
                x=base_x,
                y="Low:Q",
                tooltip=[
                    alt.Tooltip("Bulan:O", title="Bulan"),
                    alt.Tooltip("Low:Q", title="Aktual (M IDR)", format=",.3f"),
                ],
            )
        )
        layers.append(actual_point)

    return (
        alt.layer(*layers)
        .properties(height=420, title="Low Historis & Prediksi Bulanan BTC/IDR")
        .interactive()
    )


def backtest_chart(results_df):
    """
    Chart walk-forward backtest: actual vs predicted + pita CI.

    results_df kolom: month, actual_low, predicted_low, ci_lower, ci_upper.
    Returns alt.Chart interaktif.
    """
    valid = results_df.dropna(subset=["predicted_low"]).copy()
    df = pd.DataFrame(
        {
            "Bulan": _month_label(valid["month"]),
            "Aktual": valid["actual_low"] / _SCALE,
            "Prediksi": valid["predicted_low"] / _SCALE,
            "CI_Lower": valid["ci_lower"] / _SCALE,
            "CI_Upper": valid["ci_upper"] / _SCALE,
        }
    )

    base_x = alt.X("Bulan:O", title="Bulan", axis=alt.Axis(labelAngle=-45))

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.15, color=CI)
        .encode(x=base_x, y=alt.Y("CI_Lower:Q", title=_Y_TITLE), y2="CI_Upper:Q")
    )

    long = df.melt(
        id_vars=["Bulan"], value_vars=["Aktual", "Prediksi"],
        var_name="Seri", value_name="Low",
    )
    lines = (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=65, filled=True), strokeWidth=2.5)
        .encode(
            x=base_x,
            y=alt.Y("Low:Q", title=_Y_TITLE),
            color=alt.Color(
                "Seri:N",
                scale=alt.Scale(
                    domain=["Aktual", "Prediksi"], range=[ACTUAL, PRED]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.Seri == "Prediksi", alt.value([6, 4]), alt.value([0])
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Seri:N", title="Seri"),
                alt.Tooltip("Low:Q", title="Low (M IDR)", format=",.3f"),
            ],
        )
    )

    return (
        alt.layer(band, lines)
        .properties(height=420, title="Rolling Backtest - Aktual vs Prediksi")
        .interactive()
    )


def price_history_chart(df, column, period_col="Time", freq_label="Harian"):
    """
    Chart eksplorasi harga (harian / bulanan).

    Parameters:
        df: DataFrame dengan kolom period_col + kolom harga.
        column: nama kolom harga yang diplot (Low/Close/Open/High).
        period_col: nama kolom waktu.
        freq_label: 'Harian' atau 'Bulanan' untuk judul.

    Returns alt.Chart interaktif.
    """
    plot_df = pd.DataFrame(
        {
            "Periode": pd.to_datetime(df[period_col]),
            "Harga": df[column] / _SCALE,
        }
    )
    return (
        alt.Chart(plot_df)
        .mark_line(color=HIST, strokeWidth=1.6)
        .encode(
            x=alt.X("Periode:T", title="Periode"),
            y=alt.Y("Harga:Q", title=f"{column} (Miliar IDR)"),
            tooltip=[
                alt.Tooltip("Periode:T", title="Tanggal"),
                alt.Tooltip("Harga:Q", title=f"{column} (M IDR)", format=",.3f"),
            ],
        )
        .properties(height=380, title=f"BTC/IDR {column} - {freq_label}")
        .interactive()
    )


def error_bar_chart(eval_df):
    """
    Bar error% per bulan; warna hijau (prediksi >= aktual) / merah (kurang).

    eval_df kolom: Target_Month, Actual_Low, Predicted_Low, Error (%).
    Returns alt.Chart interaktif.
    """
    signed = eval_df["Predicted_Low"] - eval_df["Actual_Low"]
    df = pd.DataFrame(
        {
            "Bulan": _month_label(eval_df["Target_Month"]),
            "Error": eval_df["Error (%)"],
            "Arah": ["Over" if s >= 0 else "Under" for s in signed],
        }
    )
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Bulan:O", title="Bulan", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Error:Q", title="Error (%)"),
            color=alt.Color(
                "Arah:N",
                scale=alt.Scale(domain=["Over", "Under"], range=[POS, NEG]),
                legend=alt.Legend(title="Arah error", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Error:Q", title="Error (%)", format=".2f"),
            ],
        )
        .properties(height=320, title="Error Prediksi per Bulan")
        .interactive()
    )


def actual_vs_predicted_chart(eval_df):
    """
    Garis aktual vs prediksi untuk halaman Evaluasi.

    eval_df kolom: Target_Month, Actual_Low, Predicted_Low.
    Returns alt.Chart interaktif.
    """
    df = pd.DataFrame(
        {
            "Bulan": _month_label(eval_df["Target_Month"]),
            "Aktual": eval_df["Actual_Low"] / _SCALE,
            "Prediksi": eval_df["Predicted_Low"] / _SCALE,
        }
    )
    long = df.melt(
        id_vars=["Bulan"], value_vars=["Aktual", "Prediksi"],
        var_name="Seri", value_name="Low",
    )
    return (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5)
        .encode(
            x=alt.X("Bulan:O", title="Bulan", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Low:Q", title=_Y_TITLE),
            color=alt.Color(
                "Seri:N",
                scale=alt.Scale(
                    domain=["Aktual", "Prediksi"], range=[ACTUAL, PRED]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Seri:N", title="Seri"),
                alt.Tooltip("Low:Q", title="Low (M IDR)", format=",.3f"),
            ],
        )
        .properties(height=380, title="Aktual vs Prediksi")
        .interactive()
    )


# ============================================================================
# Chart DCA
# ============================================================================

# Warna per strategi DCA (konsisten dengan dca.py STRATEGY_COLORS)
_DCA_DOMAIN = [
    "Awal Bulan (Open)",
    "Akhir Bulan (Close)",
    "Low Bulanan (Ideal)",
    "Prediksi Model",
]
_DCA_RANGE = ["#2196F3", "#78909C", "#4CAF50", "#E91E63"]


def dca_average_cost_chart(long_df):
    """
    Line chart average cost per bulan untuk setiap strategi DCA.

    long_df: output dari dca.build_long_df(), kolom:
        Bulan (str YYYY-MM), Label (str), Average_Cost (float Rp).
    Baris dengan Average_Cost NaN (belum ada beli sama sekali) di-drop
    agar semua garis mulai dari titik yang sama-sama valid.
    Returns alt.Chart interaktif.
    """
    df = long_df.dropna(subset=["Average_Cost"]).copy()
    df["Avg_Cost_M"] = df["Average_Cost"] / _SCALE

    return (
        alt.Chart(df)
        .mark_line(
            point=alt.OverlayMarkDef(size=55, filled=True),
            strokeWidth=2.2,
        )
        .encode(
            x=alt.X(
                "Bulan:O",
                title="Bulan",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("Avg_Cost_M:Q", title="Average Cost (Miliar IDR/BTC)"),
            color=alt.Color(
                "Label:N",
                scale=alt.Scale(domain=_DCA_DOMAIN, range=_DCA_RANGE),
                legend=alt.Legend(title="Strategi", orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.Label == "Prediksi Model",
                alt.value([6, 3]),
                alt.value([0]),
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Label:N", title="Strategi"),
                alt.Tooltip(
                    "Avg_Cost_M:Q", title="Avg Cost (M IDR)", format=",.4f"
                ),
            ],
        )
        .properties(
            height=420, title="Average Cost per Bulan — Perbandingan Strategi DCA"
        )
        .interactive()
    )


def dca_cumulative_btc_chart(long_df):
    """
    Line chart BTC kumulatif terkumpul per bulan per strategi.

    long_df: output dari dca.build_long_df(), kolom Bulan, Label, Total_BTC.
    Returns alt.Chart interaktif.
    """
    df = long_df.dropna(subset=["Total_BTC"]).copy()
    df["BTC"] = df["Total_BTC"]

    return (
        alt.Chart(df)
        .mark_line(
            point=alt.OverlayMarkDef(size=55, filled=True),
            strokeWidth=2.2,
        )
        .encode(
            x=alt.X(
                "Bulan:O",
                title="Bulan",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("BTC:Q", title="Total BTC Terkumpul"),
            color=alt.Color(
                "Label:N",
                scale=alt.Scale(domain=_DCA_DOMAIN, range=_DCA_RANGE),
                legend=alt.Legend(title="Strategi", orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.Label == "Prediksi Model",
                alt.value([6, 3]),
                alt.value([0]),
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Label:N", title="Strategi"),
                alt.Tooltip("BTC:Q", title="Total BTC", format=",.8f"),
                alt.Tooltip(
                    "Total_Invested:Q", title="Total Diinvestasikan",
                    format=",.0f",
                ),
            ],
        )
        .properties(
            height=420, title="BTC Kumulatif Terkumpul — Perbandingan Strategi DCA"
        )
        .interactive()
    )


def dca_price_comparison_chart(long_df):
    """
    Line chart harga beli keempat strategi DCA per bulan dalam satu chart.

    Menampilkan harga yang dipakai (atau ditarget) tiap bulan per strategi,
    termasuk bulan miss untuk strategi model (harga prediksi tetap tampil,
    tapi lebih transparan).
    long_df: output dari dca.build_long_df().
    Returns alt.Chart interaktif.
    """
    df = long_df.dropna(subset=["Harga_Beli"]).copy()
    df["Harga_M"] = df["Harga_Beli"] / _SCALE
    df["Keterangan"] = df["Status"].fillna("Beli") if "Status" in df.columns else "Beli"

    return (
        alt.Chart(df)
        .mark_line(
            point=alt.OverlayMarkDef(size=65, filled=True),
            strokeWidth=2.2,
        )
        .encode(
            x=alt.X(
                "Bulan:O",
                title="Bulan",
                axis=alt.Axis(labelAngle=-45),
            ),
            y=alt.Y("Harga_M:Q", title="Harga Beli (Miliar IDR)"),
            color=alt.Color(
                "Label:N",
                scale=alt.Scale(domain=_DCA_DOMAIN, range=_DCA_RANGE),
                legend=alt.Legend(title="Strategi", orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.Label == "Prediksi Model",
                alt.value([6, 3]),
                alt.value([0]),
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Label:N", title="Strategi"),
                alt.Tooltip("Harga_M:Q", title="Harga (M IDR)", format=",.4f"),
                alt.Tooltip("Keterangan:N", title="Status"),
            ],
        )
        .properties(
            height=420,
            title="Perbandingan Harga Beli per Bulan — 4 Strategi DCA",
        )
        .interactive()
    )


def dca_hit_miss_chart(dca_result):
    """
    Bar chart status Hit/Miss/Skip prediksi model per bulan.

    Tiap bulan satu bar setinggi harga eksekusi (Miliar IDR), diwarnai
    sesuai hasil prediksi:
        Hit  (hijau)   — model berhasil beli di harga prediksi
        Miss (merah)   — prediksi terlalu rendah, fallback ke Close
        Skip (abu-abu) — tidak ada prediksi bulan itu

    Mengikuti pola error_bar_chart (sumbu Y numerik nyata).
    dca_result: dict output simulate_dca(), kunci 'model' harus ada.
    Returns alt.Chart atau None bila data tidak tersedia.
    """
    if "model" not in dca_result:
        return None

    src = pd.DataFrame(dca_result["model"]).copy()
    if "Status" not in src.columns or src.empty:
        return None

    def _hasil(s):
        s = str(s)
        if s.startswith("Hit"):
            return "Hit"
        if s.startswith("Miss"):
            return "Miss"
        return "Skip"

    harga = src["Harga_Beli"].fillna(0.0)
    df = pd.DataFrame(
        {
            "Bulan": _month_label(src["Bulan"]),
            "Harga": harga / _SCALE,
            "Hasil": [_hasil(s) for s in src["Status"]],
        }
    )

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Bulan:O", title="Bulan", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Harga:Q", title="Harga Eksekusi (Miliar IDR)"),
            color=alt.Color(
                "Hasil:N",
                scale=alt.Scale(
                    domain=["Hit", "Miss", "Skip"],
                    range=["#4CAF50", "#E91E63", "#9E9E9E"],
                ),
                legend=alt.Legend(title="Status", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Bulan:O", title="Bulan"),
                alt.Tooltip("Hasil:N", title="Status"),
                alt.Tooltip("Harga:Q", title="Harga (M IDR)", format=",.4f"),
            ],
        )
        .properties(height=320, title="Status Prediksi Model per Bulan")
        .interactive()
    )
