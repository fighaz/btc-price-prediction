"""
CRUD operations untuk database
"""
from datetime import date, datetime
import pandas as pd
from sqlalchemy import desc

from db.models import PriceHistory, ModelRun, Prediction, Evaluation


# ============================================================================
# Price History
# ============================================================================

def save_price_history(session, df):
    """Simpan/update data harga ke DB. Returns jumlah baris baru."""
    count = 0
    for _, row in df.iterrows():
        d = row["Time"].date() if hasattr(row["Time"], "date") else row["Time"]
        existing = session.query(PriceHistory).filter_by(time=d).first()
        if existing is None:
            session.add(PriceHistory(
                time=d,
                open=row["Open"],
                high=row["High"],
                low=row["Low"],
                close=row["Close"],
                volume=row["Volume"],
            ))
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

    data = [{
        "Time": r.time,
        "Open": r.open,
        "High": r.high,
        "Low": r.low,
        "Close": r.close,
        "Volume": r.volume,
    } for r in rows]
    df = pd.DataFrame(data)
    df["Time"] = pd.to_datetime(df["Time"])
    return df


def get_latest_price_date(session):
    """Returns tanggal terakhir di price_history atau None."""
    row = session.query(PriceHistory).order_by(desc(PriceHistory.time)).first()
    return row.time if row else None


# ============================================================================
# Model Runs
# ============================================================================

def save_model_run(session, params):
    """
    Simpan model run baru. Returns run_id.

    params keys: mode, train_end_date, forecast_days, p_lags, q_lags,
                 min_lag, alpha, n_features, train_r2, residual_std
    """
    run = ModelRun(
        mode=params["mode"],
        train_end_date=params["train_end_date"],
        forecast_days=params["forecast_days"],
        p_lags=params["p_lags"],
        q_lags=params["q_lags"],
        min_lag=params["min_lag"],
        alpha=params["alpha"],
        n_features=params["n_features"],
        train_r2=params.get("train_r2"),
        residual_std=params.get("residual_std"),
        status="running",
    )
    session.add(run)
    session.commit()
    return run.id


def update_model_run_status(session, run_id, status, metrics=None):
    """Update status dan optional metrics evaluasi."""
    run = session.query(ModelRun).get(run_id)
    if run is None:
        return
    run.status = status
    if metrics:
        run.eval_mape = metrics.get("mape")
        run.eval_rmse = metrics.get("rmse")
        run.eval_mae = metrics.get("mae")
        run.eval_r2 = metrics.get("r2")
        run.eval_ci_coverage = metrics.get("ci_coverage")
    session.commit()


def get_model_runs(session, limit=20):
    """Get model runs terbaru. Returns list of dicts."""
    rows = session.query(ModelRun).order_by(desc(ModelRun.run_at)).limit(limit).all()
    return [{
        "id": r.id,
        "run_at": r.run_at,
        "mode": r.mode,
        "train_end_date": r.train_end_date,
        "forecast_days": r.forecast_days,
        "p_lags": r.p_lags,
        "q_lags": r.q_lags,
        "min_lag": r.min_lag,
        "alpha": r.alpha,
        "n_features": r.n_features,
        "train_r2": r.train_r2,
        "status": r.status,
        "eval_mape": r.eval_mape,
        "eval_rmse": r.eval_rmse,
        "eval_ci_coverage": r.eval_ci_coverage,
    } for r in rows]


# ============================================================================
# Predictions
# ============================================================================

def save_predictions(session, run_id, forecast_df):
    """Simpan hasil prediksi dari forecast DataFrame."""
    for _, row in forecast_df.iterrows():
        session.add(Prediction(
            model_run_id=run_id,
            date=row["Time"].date() if hasattr(row["Time"], "date") else row["Time"],
            predicted_low=row["Predicted_Low"],
            confidence_lower=row["Confidence_Lower"],
            confidence_upper=row["Confidence_Upper"],
            step=row["Step"],
        ))
    session.commit()


def get_predictions_for_run(session, run_id):
    """Get prediksi untuk model run tertentu. Returns DataFrame."""
    rows = session.query(Prediction).filter_by(model_run_id=run_id).order_by(Prediction.date).all()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([{
        "Tanggal": r.date,
        "Predicted_Low": r.predicted_low,
        "Confidence_Lower": r.confidence_lower,
        "Confidence_Upper": r.confidence_upper,
        "Step": r.step,
    } for r in rows])


def get_latest_prediction(session):
    """Get prediksi dari model run terbaru yang sukses. Returns (run_info, predictions_df)."""
    run = (session.query(ModelRun)
           .filter_by(status="success")
           .order_by(desc(ModelRun.run_at))
           .first())
    if run is None:
        return None, pd.DataFrame()

    predictions_df = get_predictions_for_run(session, run.id)
    run_info = {
        "id": run.id,
        "run_at": run.run_at,
        "train_end_date": run.train_end_date,
        "forecast_days": run.forecast_days,
        "p_lags": run.p_lags,
        "q_lags": run.q_lags,
        "alpha": run.alpha,
        "train_r2": run.train_r2,
        "eval_mape": run.eval_mape,
        "eval_ci_coverage": run.eval_ci_coverage,
    }
    return run_info, predictions_df


# ============================================================================
# Evaluations
# ============================================================================

def save_evaluation(session, run_id, merged_df):
    """Simpan hasil evaluasi dari merged DataFrame (forecast vs aktual)."""
    for _, row in merged_df.iterrows():
        error_amount = row["Predicted_Low"] - row["Actual_Low"]
        error_pct = abs(error_amount) / row["Actual_Low"] * 100
        in_ci = (row["Actual_Low"] >= row["Confidence_Lower"] and
                 row["Actual_Low"] <= row["Confidence_Upper"])

        session.add(Evaluation(
            model_run_id=run_id,
            date=row["Date"] if isinstance(row["Date"], date) else row["Date"],
            actual_low=row["Actual_Low"],
            predicted_low=row["Predicted_Low"],
            error_amount=error_amount,
            error_pct=error_pct,
            in_confidence_interval=in_ci,
        ))
    session.commit()


def get_evaluation_for_run(session, run_id):
    """Get evaluasi untuk model run tertentu. Returns DataFrame."""
    rows = session.query(Evaluation).filter_by(model_run_id=run_id).order_by(Evaluation.date).all()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([{
        "Tanggal": r.date,
        "Actual_Low": r.actual_low,
        "Predicted_Low": r.predicted_low,
        "Error (Rp)": r.error_amount,
        "Error (%)": r.error_pct,
        "Dalam CI": "Ya" if r.in_confidence_interval else "Tidak",
    } for r in rows])
