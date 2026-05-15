"""
CRUD operations untuk database (skema ARDL-ECM monthly).
"""
from datetime import date

import pandas as pd
from sqlalchemy import desc

from db.models import PriceHistory, ModelRun, MonthlyPrediction, MonthlyEvaluation


def _to_date(value):
    """Normalisasi Timestamp / datetime / date -> date."""
    if hasattr(value, "date"):
        return value.date()
    return value


def _month_start(value):
    """Konversi Timestamp bulan -> date hari pertama bulan."""
    ts = pd.Timestamp(value)
    return date(ts.year, ts.month, 1)


# ============================================================================
# Price History
# ============================================================================
def save_price_history(session, df):
    """Simpan/update data harga harian ke DB. Returns jumlah baris baru."""
    count = 0
    for _, row in df.iterrows():
        d = _to_date(row["Time"])
        existing = session.query(PriceHistory).filter_by(time=d).first()
        if existing is None:
            session.add(
                PriceHistory(
                    time=d,
                    open=row["Open"],
                    high=row["High"],
                    low=row["Low"],
                    close=row["Close"],
                    volume=row["Volume"],
                )
            )
            count += 1
    session.commit()
    return count


def get_price_history(session, from_date=None, to_date=None):
    """Query data harga dari DB. Returns DataFrame."""
    query = session.query(PriceHistory).order_by(PriceHistory.time)
    if from_date:
        query = query.filter(PriceHistory.time >= from_date)
    if to_date:
        query = query.filter(PriceHistory.time <= to_date)

    rows = query.all()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            {
                "Time": r.time,
                "Open": r.open,
                "High": r.high,
                "Low": r.low,
                "Close": r.close,
                "Volume": r.volume,
            }
            for r in rows
        ]
    )
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def get_latest_price_date(session):
    """Returns tanggal terakhir di price_history atau None."""
    row = session.query(PriceHistory).order_by(desc(PriceHistory.time)).first()
    return row.time if row else None


# ============================================================================
# Model Runs
# ============================================================================
def save_model_run(session, config):
    """
    Simpan model run baru dengan config lengkap. Returns run_id.

    config keys (wajib): mode, train_end_date.
    config keys (opsional): eval_end_date, log_transform, rolling_window_years,
        max_lag_endog, max_lag_exog, ic, trend, var_maxlag, backtest_months.
    """
    run = ModelRun(
        mode=config["mode"],
        train_end_date=_to_date(config["train_end_date"]),
        eval_end_date=_to_date(config["eval_end_date"])
        if config.get("eval_end_date")
        else None,
        log_transform=config.get("log_transform", True),
        rolling_window_years=config.get("rolling_window_years", 0),
        max_lag_endog=config.get("max_lag_endog", 3),
        max_lag_exog=config.get("max_lag_exog", 3),
        ic=config.get("ic", "aic"),
        trend=config.get("trend", "c"),
        var_maxlag=config.get("var_maxlag", 3),
        backtest_months=config.get("backtest_months"),
        status="running",
    )
    session.add(run)
    session.commit()
    return run.id


def update_model_run_result(session, run_id, ardl_info=None, metrics=None,
                            status=None):
    """Update hasil model (ardl_info), metrik agregat, dan/atau status."""
    run = session.get(ModelRun, run_id)
    if run is None:
        return
    if ardl_info:
        run.endog_lag = ardl_info.get("endog_lag")
        run.exog_orders = ardl_info.get("exog_orders")
        run.bounds_f_stat = ardl_info.get("bounds_f_stat")
        run.cointegration = ardl_info.get("cointegration")
        run.lambda_ecm = ardl_info.get("lambda_ecm")
        run.half_life_months = ardl_info.get("half_life_months")
        run.train_r2 = ardl_info.get("train_r2")
    if metrics:
        run.eval_mape = metrics.get("mape")
        run.eval_rmse = metrics.get("rmse")
        run.eval_mae = metrics.get("mae")
        run.eval_r2 = metrics.get("r2")
        run.eval_ci_coverage = metrics.get("ci_coverage")
    if status:
        run.status = status
    session.commit()


def get_model_runs(session, limit=50):
    """Get model runs terbaru. Returns list of dicts."""
    rows = (
        session.query(ModelRun)
        .order_by(desc(ModelRun.run_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "run_at": r.run_at,
            "mode": r.mode,
            "train_end_date": r.train_end_date,
            "eval_end_date": r.eval_end_date,
            "log_transform": r.log_transform,
            "rolling_window_years": r.rolling_window_years,
            "max_lag_endog": r.max_lag_endog,
            "max_lag_exog": r.max_lag_exog,
            "ic": r.ic,
            "trend": r.trend,
            "var_maxlag": r.var_maxlag,
            "backtest_months": r.backtest_months,
            "endog_lag": r.endog_lag,
            "exog_orders": r.exog_orders,
            "bounds_f_stat": r.bounds_f_stat,
            "cointegration": r.cointegration,
            "lambda_ecm": r.lambda_ecm,
            "half_life_months": r.half_life_months,
            "train_r2": r.train_r2,
            "eval_mape": r.eval_mape,
            "eval_rmse": r.eval_rmse,
            "eval_mae": r.eval_mae,
            "eval_r2": r.eval_r2,
            "eval_ci_coverage": r.eval_ci_coverage,
            "status": r.status,
        }
        for r in rows
    ]


def get_model_run(session, run_id):
    """Get satu model run sebagai dict, atau None."""
    runs = get_model_runs(session, limit=10_000)
    return next((r for r in runs if r["id"] == run_id), None)


# ============================================================================
# Monthly Predictions
# ============================================================================
def save_monthly_predictions(session, run_id, forecast_df):
    """
    Simpan prediksi bulanan. forecast_df berindeks bulan target dengan kolom
    Predicted_Low, CI_Lower, CI_Upper. Mendukung 1 baris (forecast) atau
    N baris (backtest).
    """
    for idx, row in forecast_df.iterrows():
        session.add(
            MonthlyPrediction(
                model_run_id=run_id,
                target_month=_month_start(idx),
                predicted_low=float(row["Predicted_Low"]),
                ci_lower=float(row["CI_Lower"])
                if pd.notnull(row.get("CI_Lower"))
                else None,
                ci_upper=float(row["CI_Upper"])
                if pd.notnull(row.get("CI_Upper"))
                else None,
            )
        )
    session.commit()


def save_backtest_predictions(session, run_id, results_df):
    """Simpan prediksi dari hasil backtest (skip iterasi yang gagal)."""
    valid = results_df.dropna(subset=["predicted_low"])
    for _, row in valid.iterrows():
        session.add(
            MonthlyPrediction(
                model_run_id=run_id,
                target_month=_month_start(row["month"]),
                predicted_low=float(row["predicted_low"]),
                ci_lower=float(row["ci_lower"])
                if pd.notnull(row["ci_lower"])
                else None,
                ci_upper=float(row["ci_upper"])
                if pd.notnull(row["ci_upper"])
                else None,
            )
        )
    session.commit()


def get_predictions_for_run(session, run_id):
    """Get prediksi untuk model run tertentu. Returns DataFrame."""
    rows = (
        session.query(MonthlyPrediction)
        .filter_by(model_run_id=run_id)
        .order_by(MonthlyPrediction.target_month)
        .all()
    )
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Target_Month": r.target_month,
                "Predicted_Low": r.predicted_low,
                "CI_Lower": r.ci_lower,
                "CI_Upper": r.ci_upper,
            }
            for r in rows
        ]
    )


def get_latest_run(session):
    """Get model run sukses terbaru + prediksinya. Returns (run_info, pred_df)."""
    run = (
        session.query(ModelRun)
        .filter_by(status="success")
        .order_by(desc(ModelRun.run_at))
        .first()
    )
    if run is None:
        return None, pd.DataFrame()

    return get_model_run(session, run.id), get_predictions_for_run(session, run.id)


# ============================================================================
# Monthly Evaluations
# ============================================================================
def save_single_evaluation(session, run_id, eval_result):
    """Simpan evaluasi forecast tunggal (mode forecast_and_evaluate)."""
    session.add(
        MonthlyEvaluation(
            model_run_id=run_id,
            target_month=_month_start(eval_result["target_month"]),
            actual_low=eval_result["actual_low"],
            predicted_low=eval_result["predicted_low"],
            error_amount=eval_result["error_rp"],
            error_pct=eval_result["error_pct"],
            in_confidence_interval=eval_result["in_ci"],
        )
    )
    session.commit()


def save_backtest_evaluations(session, run_id, results_df):
    """Simpan evaluasi per bulan dari hasil backtest."""
    valid = results_df.dropna(subset=["predicted_low"])
    for _, row in valid.iterrows():
        error_amount = row["predicted_low"] - row["actual_low"]
        session.add(
            MonthlyEvaluation(
                model_run_id=run_id,
                target_month=_month_start(row["month"]),
                actual_low=float(row["actual_low"]),
                predicted_low=float(row["predicted_low"]),
                error_amount=float(error_amount),
                error_pct=float(row["error_pct"]),
                in_confidence_interval=bool(row["in_ci"]),
            )
        )
    session.commit()


def get_evaluation_for_run(session, run_id):
    """Get evaluasi untuk model run tertentu. Returns DataFrame."""
    rows = (
        session.query(MonthlyEvaluation)
        .filter_by(model_run_id=run_id)
        .order_by(MonthlyEvaluation.target_month)
        .all()
    )
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Target_Month": r.target_month,
                "Actual_Low": r.actual_low,
                "Predicted_Low": r.predicted_low,
                "Error (Rp)": r.error_amount,
                "Error (%)": r.error_pct,
                "Dalam CI": "Ya" if r.in_confidence_interval else "Tidak",
            }
            for r in rows
        ]
    )
