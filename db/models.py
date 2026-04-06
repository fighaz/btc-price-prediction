"""
SQLAlchemy ORM table definitions
"""
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Date, DateTime, Boolean, ForeignKey
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
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=datetime.utcnow)
    mode = Column(String, nullable=False)  # 'forecast_only' atau 'forecast_and_evaluate'
    train_end_date = Column(Date, nullable=False)
    forecast_days = Column(Integer, nullable=False)
    p_lags = Column(Integer, nullable=False)
    q_lags = Column(Integer, nullable=False)
    min_lag = Column(Integer, nullable=False)
    alpha = Column(Float, nullable=False)
    n_features = Column(Integer, nullable=False)
    train_r2 = Column(Float)
    residual_std = Column(Float)
    status = Column(String, default="running")  # running, success, failed

    # Evaluation fields (NULL jika belum dievaluasi)
    eval_mape = Column(Float)
    eval_rmse = Column(Float)
    eval_mae = Column(Float)
    eval_r2 = Column(Float)
    eval_ci_coverage = Column(Float)

    predictions = relationship("Prediction", back_populates="model_run", cascade="all, delete-orphan")
    evaluations = relationship("Evaluation", back_populates="model_run", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    predicted_low = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=False)
    confidence_upper = Column(Float, nullable=False)
    step = Column(Integer, nullable=False)

    model_run = relationship("ModelRun", back_populates="predictions")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    actual_low = Column(Float, nullable=False)
    predicted_low = Column(Float, nullable=False)
    error_amount = Column(Float, nullable=False)
    error_pct = Column(Float, nullable=False)
    in_confidence_interval = Column(Boolean, nullable=False)

    model_run = relationship("ModelRun", back_populates="evaluations")
