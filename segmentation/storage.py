"""
Storage abstraction layer.

Switches between local FileSystemStorage and Supabase Storage
based on the USE_SUPABASE_STORAGE setting (tied to DEBUG=False + SUPABASE_URL).
"""

import os
import shutil
from pathlib import Path

from django.conf import settings
from django.core.files.storage import default_storage


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class StorageBackend:
    """Abstract base for file storage operations."""

    def upload(self, local_path, remote_key):
        """Upload a local file to storage. Returns the public URL."""
        raise NotImplementedError

    def upload_content(self, content_bytes, remote_key, content_type='application/octet-stream'):
        """Upload raw bytes to storage. Returns the public URL."""
        raise NotImplementedError

    def download(self, remote_key, local_path):
        """Download a file from storage to a local path."""
        raise NotImplementedError

    def delete(self, remote_key):
        """Delete a single file from storage."""
        raise NotImplementedError

    def delete_prefix(self, prefix):
        """Delete all files under a prefix/folder."""
        raise NotImplementedError

    def get_public_url(self, remote_key):
        """Return a publicly accessible URL for the file."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Local filesystem storage
# ---------------------------------------------------------------------------

class LocalStorage(StorageBackend):
    """Stores files under MEDIA_ROOT for local development."""

    def __init__(self):
        self.media_root = Path(settings.MEDIA_ROOT)

    def upload(self, local_path, remote_key):
        dest = self.media_root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))
        return f'/media/{remote_key}'

    def upload_content(self, content_bytes, remote_key, content_type='application/octet-stream'):
        dest = self.media_root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content_bytes)
        return f'/media/{remote_key}'

    def download(self, remote_key, local_path):
        src = self.media_root / remote_key
        if not src.exists():
            raise FileNotFoundError(f'Local file not found: {src}')
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))

    def delete(self, remote_key):
        target = self.media_root / remote_key
        if target.exists():
            target.unlink()

    def delete_prefix(self, prefix):
        target_dir = self.media_root / prefix
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(str(target_dir), ignore_errors=True)

    def get_public_url(self, remote_key):
        return f'/media/{remote_key}'


# ---------------------------------------------------------------------------
# Supabase storage
# ---------------------------------------------------------------------------

class SupabaseStorage(StorageBackend):
    """Stores files in a Supabase Storage bucket for production."""

    def __init__(self):
        try:
            from supabase import create_client
        except ImportError:
            raise RuntimeError(
                'supabase package is required for production storage. '
                'Install it with: pip install supabase'
            )

        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        self.bucket = settings.SUPABASE_BUCKET

        if not self.url or not self.key:
            raise RuntimeError(
                'SUPABASE_URL and SUPABASE_KEY must be set for production storage.'
            )

        self.client = create_client(self.url, self.key)

    def upload(self, local_path, remote_key):
        with open(local_path, 'rb') as f:
            content = f.read()
        return self.upload_content(content, remote_key)

    def upload_content(self, content_bytes, remote_key, content_type='application/octet-stream'):
        storage = self.client.storage.from_(self.bucket)

        # Remove existing file if present (upsert)
        try:
            storage.remove([remote_key])
        except Exception:
            pass

        storage.upload(
            path=remote_key,
            file=content_bytes,
            file_options={'content-type': content_type},
        )

        return self.get_public_url(remote_key)

    def download(self, remote_key, local_path):
        storage = self.client.storage.from_(self.bucket)
        data = storage.download(remote_key)

        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def delete(self, remote_key):
        try:
            storage = self.client.storage.from_(self.bucket)
            storage.remove([remote_key])
        except Exception:
            pass

    def delete_prefix(self, prefix):
        try:
            storage = self.client.storage.from_(self.bucket)
            files = storage.list(prefix)
            if files:
                keys = [f'{prefix}/{f["name"]}' for f in files if f.get('name')]
                if keys:
                    storage.remove(keys)
        except Exception:
            pass

    def get_public_url(self, remote_key):
        storage = self.client.storage.from_(self.bucket)
        result = storage.get_public_url(remote_key)
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_storage_instance = None


def get_storage():
    """Return the appropriate storage backend based on settings."""
    global _storage_instance

    if _storage_instance is not None:
        return _storage_instance

    use_supabase = getattr(settings, 'USE_SUPABASE_STORAGE', False)

    if use_supabase:
        _storage_instance = SupabaseStorage()
    else:
        _storage_instance = LocalStorage()

    return _storage_instance


def storage_key_for_job(user_id, job_id, category, filename):
    """
    Build a consistent storage key for a job file.

    Categories: 'uploads', 'stacked', 'results', 'preview'
    Example: user_123/job_abc123/uploads/t1.nii.gz
    """
    safe_user_id = str(user_id or 'anonymous')
    return f'user_{safe_user_id}/job_{job_id}/{category}/{filename}'
