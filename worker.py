#!/usr/bin/env python
"""
Background worker for processing segmentation jobs.

This is a standalone process that polls the database for pending jobs
and processes them using the segmentation pipeline. Run as a separate
Railway service with: python worker.py

Environment:
    WORKER_POLL_INTERVAL  — seconds between polls (default: 3)
    WORKER_MAX_RETRIES    — max retries per job on transient failure (default: 2)
"""

import os
import sys
import time
import logging
import signal

# Ensure Django settings are loaded
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import django
django.setup()

from segmentation.models import SegmentationJob
from segmentation.tasks import process_job
from segmentation.model_loader import get_model

logger = logging.getLogger('segmentation.worker')

# Configuration
POLL_INTERVAL = int(os.environ.get('WORKER_POLL_INTERVAL', '3'))
MAX_RETRIES = int(os.environ.get('WORKER_MAX_RETRIES', '2'))

# Graceful shutdown
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    logger.info('Received signal %s — shutting down gracefully...', signum)
    _shutdown = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def preload_model():
    """Load the ML model once at worker startup (singleton pattern)."""
    try:
        logger.info('Pre-loading segmentation model...')
        model = get_model()
        logger.info('Model loaded successfully: %s', type(model).__name__)
    except Exception as exc:
        logger.warning('Model pre-load failed (will retry on first job): %s', exc)


def pick_next_job():
    """
    Atomically claim the next pending job.

    Uses select_for_update to prevent race conditions if multiple
    workers are running (unlikely on Railway free tier but safe).
    """
    from django.db import transaction

    with transaction.atomic():
        job = (
            SegmentationJob.objects
            .select_for_update(skip_locked=True)
            .filter(status='pending')
            .order_by('created_at')
            .first()
        )
        if job:
            job.status = 'processing'
            job.save(update_fields=['status', 'updated_at'])
        return job


def run_worker():
    """Main worker loop."""
    logger.info('=== BraTS Segmentation Worker Started ===')
    logger.info('Poll interval: %ds | Max retries: %d', POLL_INTERVAL, MAX_RETRIES)

    # Pre-load model at startup
    preload_model()

    while not _shutdown:
        try:
            job = pick_next_job()

            if job is None:
                time.sleep(POLL_INTERVAL)
                continue

            logger.info('Picked up job %s (created: %s)', job.id, job.created_at)

            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    process_job(job)
                    logger.info('Job %s completed successfully', job.id)
                    break
                except Exception as exc:
                    retries += 1
                    if retries > MAX_RETRIES:
                        logger.error(
                            'Job %s failed after %d retries: %s',
                            job.id, MAX_RETRIES, exc,
                        )
                        # process_job already marks the job as failed
                        break
                    else:
                        logger.warning(
                            'Job %s failed (attempt %d/%d): %s — retrying...',
                            job.id, retries, MAX_RETRIES + 1, exc,
                        )
                        # Reset status for retry
                        job.status = 'processing'
                        job.error_message = ''
                        job.save(update_fields=['status', 'error_message', 'updated_at'])
                        time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.exception('Worker loop error: %s', exc)
            time.sleep(POLL_INTERVAL)

    logger.info('=== Worker shut down ===')


if __name__ == '__main__':
    run_worker()
