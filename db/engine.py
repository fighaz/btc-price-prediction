"""
SQLAlchemy engine dan session factory.

Backend (SQLite / MySQL) ditentukan oleh config.DB_URL yang dibangun dari
environment variable. Lihat .env.example.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DB_URL, DB_BACKEND, DATA_DIR


def get_engine():
    """Buat SQLAlchemy engine. Hanya backend sqlite yang butuh DATA_DIR."""
    if DB_BACKEND == "sqlite":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False, pool_pre_ping=True)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Create semua tabel jika belum ada (jalan di SQLite maupun MySQL)."""
    from db.models import Base

    engine = get_engine()
    Base.metadata.create_all(engine)
