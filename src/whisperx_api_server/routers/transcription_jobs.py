"""
Transcription Job API endpoints - Async job queue system.
"""

import json
import logging
import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from whisperx_api_server.dependencies import get_config
from whisperx_api_server.database import get_db
from whisperx_api_server.db_models import TranscriptionJob
from whisperx_api_server.schemas import (
    TranscriptionJobResponse,
    TranscriptionJobListResponse,
    TranscriptSegment,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/audio/transcriptions", tags=["transcription-jobs"])


def _job_to_response(job: TranscriptionJob) -> TranscriptionJobResponse:
    """Convert a database job to a response schema."""
    # Parse transcript JSON if present
    text = None
    segments = None
    detected_language = None

    if job.transcript:
        try:
            transcript_data = json.loads(job.transcript)
            text = transcript_data.get("text")
            detected_language = transcript_data.get("language")
            raw_segments = transcript_data.get("segments", [])
            segments = [
                TranscriptSegment(
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    text=seg.get("text", ""),
                    speaker=seg.get("speaker"),
                )
                for seg in raw_segments
            ]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse transcript JSON for job {job.id}")

    return TranscriptionJobResponse(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model=job.model,
        language=job.language,
        diarize=job.diarize,
        min_speakers=job.min_speakers,
        max_speakers=job.max_speakers,
        chunk_size=job.chunk_size,
        text=text,
        segments=segments,
        detected_language=detected_language,
        duration=job.duration,
        processing_time=job.processing_time,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
    )


@router.post(
    "/jobs",
    response_model=TranscriptionJobResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Create a new transcription job",
    description="Upload an audio file and create a transcription job. The job will be processed asynchronously.",
)
async def create_job(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: str = Form(default="small", description="Whisper model to use"),
    language: Optional[str] = Form(default=None, description="Language code (auto-detect if not specified)"),
    diarize: bool = Form(default=False, description="Enable speaker diarization"),
    min_speakers: Optional[int] = Form(default=None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Form(default=None, description="Maximum number of speakers"),
    chunk_size: int = Form(default=15, description="Chunk size in seconds"),
    vad_onset: float = Form(default=0.5, description="VAD onset threshold"),
    vad_offset: float = Form(default=0.363, description="VAD offset threshold"),
    db: Session = Depends(get_db),
):
    """Create a new transcription job."""

    config = get_config()
    upload_dir = os.getenv("UPLOAD_DIR", "/workspace/data/uploads")
    
    # Ensure upload directory exists
    os.makedirs(upload_dir, exist_ok=True)

    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    audio_path = os.path.join(upload_dir, f"{job_id}{file_ext}")

    try:
        # Save uploaded file
        contents = await file.read()
        with open(audio_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved audio file: {audio_path} ({len(contents)} bytes)")
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save audio file")

    # Create job record
    job = TranscriptionJob(
        id=job_id,
        status="pending",
        audio_path=audio_path,
        original_filename=file.filename,
        model=model,
        language=language,
        diarize=diarize,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        chunk_size=chunk_size,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info(f"Created job: {job.id}")
    return _job_to_response(job)


@router.get(
    "/jobs",
    response_model=TranscriptionJobListResponse,
    summary="List transcription jobs",
    description="Get a paginated list of all transcription jobs.",
)
async def list_jobs(
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all transcription jobs with pagination."""

    # Build query
    query = db.query(TranscriptionJob)

    if status:
        query = query.filter(TranscriptionJob.status == status)

    # Get total count
    total = query.count()

    # Apply pagination
    offset = (page - 1) * limit
    jobs = query.order_by(TranscriptionJob.created_at.desc()).offset(offset).limit(limit).all()

    return TranscriptionJobListResponse(
        jobs=[_job_to_response(job) for job in jobs],
        total=total,
        page=page,
        limit=limit,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=TranscriptionJobResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get job details",
    description="Get the details and status of a specific transcription job.",
)
async def get_job(job_id: str, db: Session = Depends(get_db)):
    """Get a specific transcription job."""

    job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return _job_to_response(job)


@router.delete(
    "/jobs/{job_id}",
    responses={404: {"model": ErrorResponse}},
    summary="Delete a job",
    description="Delete a transcription job and its associated data.",
)
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a transcription job."""

    job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Delete audio file if it still exists
    if job.audio_path and os.path.exists(job.audio_path):
        try:
            os.remove(job.audio_path)
        except Exception as e:
            logger.warning(f"Failed to delete audio file: {e}")

    # Delete job record
    db.delete(job)
    db.commit()

    logger.info(f"Deleted job: {job_id}")
    return {"message": f"Job {job_id} deleted"}
