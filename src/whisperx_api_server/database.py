"""
Database setup and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from whisperx_api_server.dependencies import get_config
import os

config = get_config()
# Database path - configurable via environment variable
DATABASE_PATH = os.getenv("DATABASE_PATH", "/workspace/data/whisperx.db")

# Create database URL
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Create engine with SQLite-specific settings
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite with threading
    echo=False,  # Set to True for SQL debugging
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency that provides a database session.
    Use with FastAPI's Depends().
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    Called on application startup.
    """
    from whisperx_api_server.db_models import TranscriptionJob  # noqa: F401

    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    Base.metadata.create_all(bind=engine)
