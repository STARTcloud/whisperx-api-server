"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# --- Request Schemas ---

class TranscriptionJobCreate(BaseModel):
    """Schema for creating a new transcription job (form data parsed separately)."""

    model: str = Field(default="small", description="Whisper model to use")
    language: Optional[str] = Field(default=None, description="Language code (auto-detected if not specified)")
    diarize: bool = Field(default=False, description="Enable speaker diarization")
    min_speakers: Optional[int] = Field(default=None, ge=1, description="Minimum number of speakers")
    max_speakers: Optional[int] = Field(default=None, ge=1, description="Maximum number of speakers")
    chunk_size: int = Field(default=15, ge=1, le=60, description="Chunk size in seconds")
    vad_onset: float = Field(default=0.5, ge=0.0, le=1.0, description="VAD onset threshold")
    vad_offset: float = Field(default=0.363, ge=0.0, le=1.0, description="VAD offset threshold")


# --- Response Schemas ---

class TranscriptSegment(BaseModel):
    """A single segment of the transcript."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text")
    speaker: Optional[str] = Field(default=None, description="Speaker label (if diarization enabled)")


class TranscriptionJobResponse(BaseModel):
    """Response schema for a transcription job."""

    id: str = Field(description="Job ID")
    status: str = Field(description="Job status: pending, processing, completed, failed")
    original_filename: Optional[str] = Field(default=None, description="Original audio filename")

    # Configuration
    model: str = Field(description="Whisper model used")
    language: Optional[str] = Field(default=None, description="Language code")
    diarize: bool = Field(description="Whether diarization is enabled")
    min_speakers: Optional[int] = Field(default=None, description="Minimum speakers")
    max_speakers: Optional[int] = Field(default=None, description="Maximum speakers")
    chunk_size: int = Field(description="Chunk size in seconds")

    # Results (only present when completed)
    text: Optional[str] = Field(default=None, description="Full transcript text")
    segments: Optional[List[TranscriptSegment]] = Field(default=None, description="Transcript segments")
    detected_language: Optional[str] = Field(default=None, description="Detected language")

    # Metadata
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")

    # Timestamps
    created_at: datetime = Field(description="Job creation time")
    updated_at: datetime = Field(description="Last update time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")

    class Config:
        from_attributes = True


class TranscriptionJobListResponse(BaseModel):
    """Response schema for listing transcription jobs."""

    jobs: List[TranscriptionJobResponse] = Field(description="List of jobs")
    total: int = Field(description="Total number of jobs")
    page: int = Field(description="Current page")
    limit: int = Field(description="Items per page")


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str = Field(description="Model name")
    description: str = Field(description="Model description")


class ModelsResponse(BaseModel):
    """Response schema for listing available models."""

    models: List[ModelInfo] = Field(description="Available models")
    default: str = Field(description="Default model")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    worker_status: str = Field(description="Background worker status")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
