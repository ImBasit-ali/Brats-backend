import io
import base64
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from urllib.parse import urlparse

import nibabel as nib
import numpy as np
from PIL import Image
from django.conf import settings
from django.core.files.base import ContentFile
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import SegmentationJob, UploadedFile
from .serializers import (
    SegmentationJobStatusSerializer,
    SegmentationJobResultSerializer,
)
from .stacking import (
    EXPECTED_MODALITIES,
    infer_extension,
)
from .storage import get_storage, storage_key_for_job
from django.http import JsonResponse

@api_view(['GET'])
def worker_health_check(request):
    """
    GET /api/health/
    Production health check endpoint — used by Railway healthcheckPath.
    """
    import logging
    from .model_loader import get_model

    logger = logging.getLogger(__name__)

    health_data = {
        'backend': 'ok',
        'model_available': False,
        'model_error': None,
        'redis_available': False,
        'redis_error': None,
        'pending_jobs': 0,
        'processing_jobs': 0,
        'sync_mode_available': True,
        'environment': 'railway' if os.environ.get('RAILWAY_ENVIRONMENT') else 'local',
    }

    try:
        health_data['pending_jobs'] = SegmentationJob.objects.filter(status='pending').count()
        health_data['processing_jobs'] = SegmentationJob.objects.filter(status='processing').count()
    except Exception as db_exc:
        health_data['db_error'] = str(db_exc)

    from .model_loader import MODEL as _CURRENT_MODEL
    if _CURRENT_MODEL is not None:
        health_data['model_available'] = True
    else:
        health_data['model_error'] = 'Model not yet loaded'

    return Response(health_data, status=status.HTTP_200_OK)


def _get_user_id(request):
    """Get or create an anonymous user ID from the session."""
    if hasattr(request, 'session'):
        user_id = request.session.get('anon_user_id')
        if not user_id:
            user_id = uuid4().hex[:16]
            request.session['anon_user_id'] = user_id
        return user_id
    return uuid4().hex[:16]


def _build_public_url(request, path):
    url = request.build_absolute_uri(path)
    if os.environ.get('RAILWAY_ENVIRONMENT') and url.startswith('http://'):
        return f"https://{url[len('http://'):]}"
    return url


def _upload_request_file_to_storage(uploaded_file, storage, remote_key):
    suffix = infer_extension(uploaded_file.name) or '.bin'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name

    try:
        with open(temp_path, 'wb') as target:
            if hasattr(uploaded_file, 'chunks'):
                for chunk in uploaded_file.chunks():
                    target.write(chunk)
            else:
                target.write(uploaded_file.read())

        return storage.upload(temp_path, remote_key)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def _get_file_wrappers_from_job(job):
    """Helper to download files from storage and return them as file wrappers."""
    storage = get_storage()
    file_wrappers = []
    
    if not job.input_files_json:
        raise ValueError("No input files associated with this job.")
        
    for item in job.input_files_json:
        key = item['key']
        suffix = infer_extension(item['original_name']) or '.nii'
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        storage.download(key, tmp.name)
        
        wrapper = SimpleNamespace(
            file=None,
            temporary_file_path=lambda p=tmp.name: p,
            path=tmp.name,
            original_name=item['original_name'],
            modality=item['modality']
        )
        file_wrappers.append(wrapper)
        
    return file_wrappers

def _cleanup_wrappers(file_wrappers):
    """Helper to cleanup temporary downloaded files."""
    for w in file_wrappers:
        try:
            os.unlink(w.path)
        except OSError:
            pass


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_draft_files(request):
    """
    POST /api/segment/upload/
    Upload NIfTI files ONCE. Deletes previous jobs/files for this user.
    Returns a job_id (status: draft).
    """
    files = request.FILES.getlist('files')
    modalities = request.POST.getlist('modalities')

    if not files:
        return Response(
            {'error': 'No files uploaded. Please upload at least one NIfTI file.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    user_id = _get_user_id(request)
    storage = get_storage()

    # --- CLEANUP PREVIOUS JOBS FOR THIS USER ---
    # Delete previous jobs to conserve storage (as requested by user)
    old_jobs = SegmentationJob.objects.filter(user_id=user_id)
    for old_job in old_jobs:
        job_prefix = f'user_{user_id}/job_{old_job.id}'
        try:
            storage.delete_prefix(f'{job_prefix}/uploads')
            storage.delete_prefix(f'{job_prefix}/stacked')
            storage.delete_prefix(f'{job_prefix}/results')
            storage.delete_prefix(f'{job_prefix}/preview')
        except Exception:
            pass
        old_job.delete()

    # --- CREATE NEW DRAFT JOB ---
    job = SegmentationJob.objects.create(
        user_id=user_id,
        status='draft',
        grade='HGG',
        regions={},
        opacity=70
    )

    # Save uploaded files
    input_keys = []
    for i, file in enumerate(files):
        if len(files) == 4:
            modality = modalities[i] if i < len(modalities) else EXPECTED_MODALITIES[i]
        else:
            modality = 'stacked'

        safe_name = Path(file.name).name
        remote_key = storage_key_for_job(user_id, job.id, 'uploads', safe_name)
        upload_url = _upload_request_file_to_storage(file, storage, remote_key)

        input_keys.append(
            {
                'key': remote_key,
                'modality': modality,
                'original_name': safe_name,
                'url': upload_url,
            }
        )

        # Keep local UploadedFile records in local/dev mode for backward compatibility.
        if not settings.USE_SUPABASE_STORAGE:
            if hasattr(file, 'seek'):
                file.seek(0)
            UploadedFile.objects.create(
                job=job,
                file=file,
                original_name=file.name,
                modality=modality,
            )

    job.input_files_json = input_keys
    job.save(update_fields=['input_files_json'])

    return Response({
        'job_id': str(job.id),
        'success': True
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def create_segmentation(request):
    """
    POST /api/segment/
    Updates the draft job with settings and starts background segmentation.
    Expects job_id instead of files.
    """
    job_id = request.data.get('job_id') or request.POST.get('job_id')
    if not job_id:
        return Response({'error': 'job_id is required. Upload files first.'}, status=status.HTTP_400_BAD_REQUEST)

    job = get_object_or_404(SegmentationJob, id=job_id)

    grade = request.data.get('grade') or request.POST.get('grade') or 'HGG'
    regions_str = request.data.get('regions') or request.POST.get('regions') or '{}'
    opacity = int(request.data.get('opacity') or request.POST.get('opacity') or 70)

    try:
        regions = json.loads(regions_str) if isinstance(regions_str, str) else regions_str
    except (json.JSONDecodeError, TypeError):
        regions = {}

    job.grade = grade
    job.regions = regions
    job.opacity = opacity
    job.status = 'pending'
    job.save()

    sync_mode = str(request.data.get('sync') or request.POST.get('sync', '')).lower() in ('true', '1', 'yes')

    if sync_mode:
        try:
            from .tasks import process_job
            process_job(job)
            job.refresh_from_db()
            return Response(
                {
                    'id': str(job.id),
                    'status': job.status,
                    'created_at': job.created_at,
                    'sync': True,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
            return Response(
                {
                    'id': str(job.id),
                    'status': 'failed',
                    'error': str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    from .tasks import mock_process_segmentation
    mock_process_segmentation(job.id)

    return Response(
        {
            'id': str(job.id),
            'status': 'pending',
            'created_at': job.created_at,
            'sync': False,
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(['POST'])
def stack_preview(request):
    """
    POST /api/segment/stack/
    Loads previously uploaded files via job_id, stacks them sequentially, and returns preview.
    """
    job_id = request.data.get('job_id') or request.POST.get('job_id')
    if not job_id:
        return Response({'error': 'job_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

    job = get_object_or_404(SegmentationJob, id=job_id)
    file_wrappers = []

    try:
        file_wrappers = _get_file_wrappers_from_job(job)
        if not file_wrappers:
            raise ValueError('No files found for this job.')

        extension = infer_extension(file_wrappers[0].original_name)
        if extension not in ('.nii', '.nii.gz', '.png'):
            raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

        if extension == '.png':
            from .stacking import stack_png_files
            stacked_volume = stack_png_files(file_wrappers)
        else:
            from .stacking import stack_nifti_files
            stacked_volume = stack_nifti_files(file_wrappers)

        stacked_url = None
        if extension in ('.nii', '.nii.gz'):
            preview_dir = Path(settings.MEDIA_ROOT) / 'previews'
            preview_dir.mkdir(parents=True, exist_ok=True)
            stacked_name = f'stacked_preview_{uuid4().hex}.nii.gz'
            stacked_path = preview_dir / stacked_name
            nib.save(stacked_volume, str(stacked_path))
            stacked_relative_url = f"{settings.MEDIA_URL.rstrip('/')}/previews/{stacked_name}"
            stacked_url = _build_public_url(request, stacked_relative_url)

        if extension in ('.nii', '.nii.gz'):
            volume_data = np.asarray(stacked_volume.dataobj, dtype=np.float32)
            mid_slice_idx = volume_data.shape[2] // 2
            preview_slice = volume_data[:, :, mid_slice_idx, 0]
            
            vmin, vmax = preview_slice.min(), preview_slice.max()
            if vmax > vmin:
                preview_slice = (preview_slice - vmin) / (vmax - vmin) * 255
            preview_slice = np.clip(preview_slice, 0, 255).astype(np.uint8)
            preview_img = Image.fromarray(preview_slice, mode='L')
        else:
            preview_img = stacked_volume.split()[0]
        
        preview_bytes_io = io.BytesIO()
        preview_img.save(preview_bytes_io, format='PNG')
        preview_b64 = base64.b64encode(preview_bytes_io.getvalue()).decode('ascii')

        return Response(
            {
                'success': True,
                'status': 'stacked',
                'stacked_url': stacked_url,
                'preview': preview_b64,
                'preview_url': f'data:image/png;base64,{preview_b64}',
                'mode': 'stacked-instant',
            },
            status=status.HTTP_200_OK,
        )

    except ValueError as exc:
        return Response({'success': False, 'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return Response({'success': False, 'error': f'Stack error: {str(exc)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        _cleanup_wrappers(file_wrappers)


@api_view(['POST'])
def view_individual_uploads(request):
    """
    POST /api/segment/view-uploads/
    View individual uploaded files via job_id WITHOUT stacking.
    """
    job_id = request.data.get('job_id') or request.POST.get('job_id')
    if not job_id:
        return Response({'error': 'job_id is required.'}, status=status.HTTP_400_BAD_REQUEST)

    job = get_object_or_404(SegmentationJob, id=job_id)
    file_wrappers = []

    try:
        file_wrappers = _get_file_wrappers_from_job(job)
        if not file_wrappers:
            raise ValueError('No files found for this job.')

        extension = infer_extension(file_wrappers[0].original_name)
        if extension not in ('.nii', '.nii.gz', '.png'):
            raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

        individual_volumes = []
        
        if extension in ('.nii', '.nii.gz'):
            from .stacking import _load_nifti_file
            
            for item in file_wrappers:
                modality, data, affine, header = _load_nifti_file(item)
                
                mid_z = data.shape[2] // 2
                preview_slice = data[:, :, mid_z]
                
                vmin, vmax = float(np.min(preview_slice)), float(np.max(preview_slice))
                if vmax > vmin:
                    preview_slice = ((preview_slice - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                else:
                    preview_slice = np.zeros_like(preview_slice, dtype=np.uint8)
                
                preview_img = Image.fromarray(preview_slice, mode='L')
                preview_bytes = io.BytesIO()
                preview_img.save(preview_bytes, format='PNG')
                preview_b64 = base64.b64encode(preview_bytes.getvalue()).decode('ascii')
                
                individual_volumes.append({
                    'modality': modality,
                    'original_name': item.original_name,
                    'shape': list(data.shape),
                    'preview_url': f'data:image/png;base64,{preview_b64}',
                    'visible': True,
                })
        else:
            from .stacking import _load_png_file
            
            for item in file_wrappers:
                modality, img = _load_png_file(item)
                
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_b64 = base64.b64encode(img_bytes.getvalue()).decode('ascii')
                
                individual_volumes.append({
                    'modality': modality,
                    'original_name': item.original_name,
                    'size': list(img.size),
                    'preview_url': f'data:image/png;base64,{img_b64}',
                    'visible': True,
                })

        return Response(
            {
                'success': True,
                'status': 'individual-uploads-loaded',
                'extension': extension,
                'volumes': individual_volumes,
                'mode': 'individual-view',
                'total_volumes': len(individual_volumes),
            },
            status=status.HTTP_200_OK,
        )

    except ValueError as exc:
        return Response({'success': False, 'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return Response({'success': False, 'error': f'Upload preview error: {str(exc)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        _cleanup_wrappers(file_wrappers)


@api_view(['GET'])
def get_segmentation_status(request, job_id):
    job = get_object_or_404(SegmentationJob, id=job_id)
    serializer = SegmentationJobStatusSerializer(job, context={'request': request})
    data = serializer.data
    if job.status in ('error', 'failed'):
        data['error'] = job.error_message
    return Response(data)


@api_view(['GET'])
def get_segmentation_result(request, job_id):
    job = get_object_or_404(SegmentationJob, id=job_id)
    if job.status not in ('done', 'error', 'failed'):
        return Response({'error': 'Job is still processing.', 'status': job.status}, status=status.HTTP_202_ACCEPTED)
    serializer = SegmentationJobResultSerializer(job, context={'request': request})
    return Response(serializer.data)


@api_view(['GET'])
def download_segmentation(request, job_id):
    job = get_object_or_404(SegmentationJob, id=job_id)
    if job.status != 'done':
        return Response({'error': 'Segmentation not complete.'}, status=status.HTTP_400_BAD_REQUEST)

    if job.segmentation_file:
        from django.http import FileResponse
        return FileResponse(
            job.segmentation_file.open('rb'),
            as_attachment=True,
            filename=f'segmentation_{job.id}.nii.gz',
        )

    if not job.mask_url:
        return Response({'error': 'No segmentation file available.'}, status=status.HTTP_404_NOT_FOUND)

    parsed = urlparse(job.mask_url)
    if not parsed.scheme:
        local_path = Path(settings.MEDIA_ROOT) / job.mask_url.replace('/media/', '')
        if not local_path.exists():
            return Response({'error': 'Segmentation mask file not found.'}, status=status.HTTP_404_NOT_FOUND)
        from django.http import FileResponse
        return FileResponse(open(local_path, 'rb'), as_attachment=True, filename=f'segmentation_{job.id}.nii.gz')

    try:
        import requests
        from django.http import HttpResponse
        upstream = requests.get(job.mask_url, timeout=120)
        upstream.raise_for_status()
        response = HttpResponse(upstream.content, content_type='application/gzip')
        response['Content-Disposition'] = f'attachment; filename="segmentation_{job.id}.nii.gz"'
        return response
    except Exception as exc:
        return Response({'error': f'Failed to fetch segmentation from storage: {exc}'}, status=status.HTTP_502_BAD_GATEWAY)
