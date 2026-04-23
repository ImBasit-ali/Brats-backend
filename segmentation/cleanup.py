"""
File lifecycle management.

Keeps only the latest completed job per user. Deletes older jobs
(and their storage files) after a new job finishes successfully.
"""

import logging

from django.utils import timezone

logger = logging.getLogger(__name__)


def cleanup_old_jobs(current_job):
    """
    Delete all older jobs for the same user after the current job completes.

    Safety rules:
    - Never deletes the current job
    - Never deletes a job that is still processing
    - Only runs when the current job is 'done'
    """
    from .models import SegmentationJob
    from .storage import get_storage, storage_key_for_job

    if current_job.status != 'done':
        return

    user_id = current_job.user_id
    if not user_id:
        # No user scoping — skip cleanup
        return

    # Find all other jobs for this user
    other_jobs = SegmentationJob.objects.filter(
        user_id=user_id,
    ).exclude(
        id=current_job.id,
    )

    storage = get_storage()

    for job in other_jobs:
        if job.status == 'processing':
            logger.info(
                'Skipping cleanup of job %s — still processing', job.id
            )
            continue

        # Delete storage files
        job_prefix = f'user_{user_id}/job_{job.id}'
        try:
            storage.delete_prefix(f'{job_prefix}/uploads')
            storage.delete_prefix(f'{job_prefix}/stacked')
            storage.delete_prefix(f'{job_prefix}/results')
            storage.delete_prefix(f'{job_prefix}/preview')
        except Exception as exc:
            logger.warning(
                'Failed to delete storage files for job %s: %s', job.id, exc
            )

        # Delete database records (cascades to UploadedFile)
        logger.info('Deleting old job %s for user %s', job.id, user_id)
        job.delete()
