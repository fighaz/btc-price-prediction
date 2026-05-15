"""
SQLAlchemy ORM table definitions untuk engine ARDL-ECM monthly.

Skema portabel SQLite <-> MySQL: semua kolom String diberi panjang eksplisit.

Empat tabel:
    PriceHistory      — cache data harga harian dari Indodax
    ModelRun          — config + hasil + status tiap eksekusi pipeline
    MonthlyPrediction — history prediksi bulanan (1 baris forecast / N baris backtest)
    MonthlyEvaluation — perbandingan prediksi vs aktual
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, Date, DateTime, Boolean, ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(Date, nullable=False, unique=True, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class ModelRun(Base):
    """Satu baris per eksekusi pipeline: menyimpan config, hasil, dan status."""

    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=datetime.utcnow)

    # Mode: 'forecast' | 'forecast_and_evaluate' | 'backtest'
    mode = Column(String(30), nullable=False)

    # --- Config run (parameter yang dipilih, supaya run bisa direproduksi) ---
    train_end_date = Column(Date, nullable=False)
    eval_end_date = Column(Date)  # hanya mode forecast_and_evaluate
    log_transform = Column(Boolean, nullable=False, default=True)
    rolling_window_years = Column(Integer, nullable=False, default=0)
    max_lag_endog = Column(Integer, nullable=False, default=3)
    max_lag_exog = Column(Integer, nullable=False, default=3)
    ic = Column(String(10), nullable=False, default="aic")
    trend = Column(String(5), nullable=False, default="c")
    var_maxlag = Column(Integer, nullable=False, default=3)
    backtest_months = Column(Integer)  # NULL kalau bukan mode backtest

    # --- Hasil model (mode forecast / forecast_and_evaluate) ---
    endog_lag = Column(Integer)
    exog_orders = Column(String(255))
    bounds_f_stat = Column(Float)
    cointegration = Column(String(30))
    lambda_ecm = Column(Float)
    half_life_months = Column(Float)
    train_r2 = Column(Float)

    # --- Metrik agregat (mode evaluate / backtest) ---
    eval_mape = Column(Float)
    eval_rmse = Column(Float)
    eval_mae = Column(Float)
    eval_r2 = Column(Float)
    eval_ci_coverage = Column(Float)

    status = Column(String(20), nullable=False, default="running")

    predictions = relationship(
        "MonthlyPrediction", back_populates="model_run",
        cascade="all, delete-orphan",
    )
    evaluations = relationship(
        "MonthlyEvaluation", back_populates="model_run",
        cascade="all, delete-orphan",
    )


class MonthlyPrediction(Base):
    """History prediksi bulanan. 1 baris untuk forecast, N baris untuk backtest."""

    __tablename__ = "monthly_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(
        Integer, ForeignKey("model_runs.id"), nullable=False, index=True
    )
    target_month = Column(Date, nullable=False)  # hari pertama bulan target
    predicted_low = Column(Float, nullable=False)
    ci_lower = Column(Float)
    ci_upper = Column(Float)

    model_run = relationship("ModelRun", back_populates="predictions")


class MonthlyEvaluation(Base):
    """Perbandingan prediksi vs aktual per bulan."""

    __tablename__ = "monthly_evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(
        Integer, ForeignKey("model_runs.id"), nullable=False, index=True
    )
    target_month = Column(Date, nullable=False)
    actual_low = Column(Float, nullable=False)
    predicted_low = Column(Float, nullable=False)
    error_amount = Column(Float, nullable=False)
    error_pct = Column(Float, nullable=False)
    in_confidence_interval = Column(Boolean, nullable=False)

    model_run = relationship("ModelRun", back_populates="evaluations")
