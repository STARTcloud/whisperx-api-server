"""
SQLAlchemy database models for job persistence.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime
from sqlalchemy.sql import func

from whisperx_api_server.database import Base


class TranscriptionJob(Base):
    """
    Represents a transcription job in the database.

    Adding a column here is not enough on its own: init_db() calls create_all(), which creates
    missing *tables* and never missing *columns*. A database written by an earlier version keeps
    its old shape, and every query against the new model then fails against the deployed data.
    A new column needs an explicit ALTER on startup (or a migration tool) alongside it.
    """

    __tablename__ = "transcription_jobs"

    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Status: pending, processing, completed, failed
    status = Column(String(20), nullable=False, default="pending", index=True)

    # File information
    audio_path = Column(Text, nullable=False)
    original_filename = Column(Text, nullable=True)

    # Transcription result (JSON string)
    transcript = Column(Text, nullable=True)

    # Error message if failed
    error_message = Column(Text, nullable=True)

    # Configuration options
    model = Column(String(50), nullable=False, default="small")
    language = Column(String(10), nullable=True)
    diarize = Column(Boolean, nullable=False, default=False)
    min_speakers = Column(Integer, nullable=True)
    max_speakers = Column(Integer, nullable=True)
    chunk_size = Column(Integer, nullable=False, default=15)
    vad_onset = Column(Float, nullable=False, default=0.5)
    vad_offset = Column(Float, nullable=False, default=0.363)

    # Metadata
    duration = Column(Float, nullable=True)  # Audio duration in seconds
    processing_time = Column(Float, nullable=True)  # Processing time in seconds

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<TranscriptionJob(id={self.id}, status={self.status})>"
