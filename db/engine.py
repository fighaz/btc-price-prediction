"""
SQLAlchemy engine dan session factory
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import DB_URL, DATA_DIR


def get_engine():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(DB_URL, echo=False)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    """Create semua tabel jika belum ada"""
    from db.models import Base
    engine = get_engine()
    Base.metadata.create_all(engine)
