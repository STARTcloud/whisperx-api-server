"""
Background worker for processing transcription jobs.
"""

import json
import logging
import os
import threading
import time
import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from whisperx_api_server.database import SessionLocal
from whisperx_api_server.db_models import TranscriptionJob
from whisperx_api_server import transcriber
from whisperx_api_server.models import load_model_instance

logger = logging.getLogger(__name__)


class TranscriptionWorker:
    """
    Background worker that processes pending transcription jobs.
    Runs in a separate thread and polls for new jobs.
    """

    def __init__(self, poll_interval: float = 2.0):
        """
        Initialize the worker.

        Args:
            poll_interval: How often to check for new jobs (in seconds)
        """
        self._poll_interval = poll_interval
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._current_job_id: Optional[str] = None

    def start(self):
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Worker is already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Transcription worker started")

    def stop(self):
        """Stop the background worker thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Transcription worker stopped")

    def is_running(self) -> bool:
        """Check if the worker is running."""
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_status(self) -> str:
        """Get the worker status."""
        if not self.is_running():
            return "stopped"
        if self._current_job_id:
            return f"processing:{self._current_job_id}"
        return "idle"

    def _run(self):
        """Main worker loop."""
        logger.info("Worker loop started")

        while self._running:
            try:
                job = self._get_next_job()
                if job:
                    self._process_job(job)
                else:
                    time.sleep(self._poll_interval)
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(self._poll_interval)

        logger.info("Worker loop ended")

    def _get_next_job(self) -> Optional[TranscriptionJob]:
        """
        Get the next pending job and mark it as processing.
        """
        db = SessionLocal()
        try:
            job = (
                db.query(TranscriptionJob)
                .filter(TranscriptionJob.status == "pending")
                .order_by(TranscriptionJob.created_at.asc())
                .first()
            )

            if job:
                job.status = "processing"
                db.commit()
                db.refresh(job)
                logger.info(f"Picked up job: {job.id}")
                return job

            return None
        except Exception as e:
            db.rollback()
            logger.error(f"Error getting next job: {e}")
            return None
        finally:
            db.close()

    def _process_job(self, job: TranscriptionJob):
        """
        Process a transcription job using Nyralei's transcriber.
        """
        self._current_job_id = job.id
        db = SessionLocal()

        try:
            logger.info(f"Processing job {job.id}")
            
            # Create a temporary file-like object from the saved audio path
            class FakeUploadFile:
                def __init__(self, filepath):
                    self.filepath = filepath
                    self.filename = os.path.basename(filepath)
                    
                async def read(self):
                    with open(self.filepath, 'rb') as f:
                        return f.read()
            
            fake_file = FakeUploadFile(job.audio_path)
            
            # Run transcription using Nyralei's async transcriber in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            start_time = time.time()
            
            # Load model instance
            model_instance = loop.run_until_complete(load_model_instance(job.model))
            
            # Run transcription
            result = loop.run_until_complete(transcriber.transcribe(
                audio_file=fake_file,
                batch_size=16,
                chunk_size=job.chunk_size,
                asr_options={},
                language=job.language,
                whispermodel=model_instance,
                align=True,
                diarize=job.diarize,
                request_id=job.id,
                task="transcribe",
            ))
            
            loop.close()
            
            processing_time = time.time() - start_time

            # Update job in database
            job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job.id).first()
            
            job.status = "completed"
            
            # Extract data from result
            segments = result.get("segments", [])
            if isinstance(segments, dict):
                segments = segments.get("segments", [])
                
            job.transcript = json.dumps({
                "text": result.get("text", ""),
                "segments": segments,
                "language": result.get("language", ""),
            })
            job.duration = result.get("duration")
            job.processing_time = processing_time
            job.completed_at = datetime.utcnow()
            logger.info(f"Job {job.id} completed successfully in {processing_time:.2f}s")

            db.commit()

            # Delete audio file after processing
            self._cleanup_audio_file(job.audio_path)

        except Exception as e:
            logger.error(f"Error processing job {job.id}: {e}", exc_info=True)

            # Update job status to failed
            try:
                job = db.query(TranscriptionJob).filter(TranscriptionJob.id == job.id).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    job.processing_time = time.time() - start_time if 'start_time' in locals() else None
                    db.commit()

                    # Still try to clean up audio file
                    self._cleanup_audio_file(job.audio_path)
            except Exception as update_error:
                logger.error(f"Error updating failed job: {update_error}")
                db.rollback()
        finally:
            db.close()
            self._current_job_id = None

    def _cleanup_audio_file(self, audio_path: str):
        """
        Delete the audio file after processing.
        """
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Deleted audio file: {audio_path}")
        except Exception as e:
            logger.warning(f"Failed to delete audio file {audio_path}: {e}")


# Global worker instance
worker = TranscriptionWorker()
