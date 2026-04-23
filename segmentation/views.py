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


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def create_segmentation(request):
    """
    POST /api/segment/
    Upload NIfTI files and create a new segmentation job.

    Returns job_id immediately — processing happens in the background worker.
    """
    files = request.FILES.getlist('files')
    modalities = request.POST.getlist('modalities')

    if not files:
        return Response(
            {'error': 'No files uploaded. Please upload at least one NIfTI file.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Create job (no model loading in request path)
    grade = request.POST.get('grade', 'HGG')
    regions_str = request.POST.get('regions', '{}')
    opacity = int(request.POST.get('opacity', 70))

    try:
        regions = json.loads(regions_str) if isinstance(regions_str, str) else regions_str
    except (json.JSONDecodeError, TypeError):
        regions = {}

    # Get or create anonymous user_id from session
    user_id = _get_user_id(request)

    job = SegmentationJob.objects.create(
        grade=grade,
        regions=regions,
        opacity=opacity,
        user_id=user_id,
        status='pending',
    )

    storage = get_storage()

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

    # Return immediately — worker picks up the job
    return Response(
        {
            'id': str(job.id),
            'status': job.status,
            'created_at': job.created_at,
        },
        status=status.HTTP_201_CREATED,
    )


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def stack_preview(request):
    """
    POST /api/segment/stack/
    Stack the uploaded modalities and return the full stacked NIfTI volume URL
    plus a quick preview image for fast UI feedback.
    """
    files = request.FILES.getlist('files')
    modalities = request.POST.getlist('modalities')

    if not files:
        return Response(
            {'success': False, 'error': 'No files uploaded. Please upload at least one NIfTI file.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    temp_paths = []
    try:
        # Determine extension from first file
        extension = infer_extension(files[0].name)
        if extension not in ('.nii', '.nii.gz', '.png'):
            raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

        user_id = _get_user_id(request)
        storage = get_storage()

        # Create wrapper objects for stacking
        file_wrappers = []
        for index, file_obj in enumerate(files):
            modality = modalities[index] if index < len(modalities) else (EXPECTED_MODALITIES[index] if index < 4 else None)
            if not modality:
                raise ValueError(f'Unable to determine modality for file {index}')
            wrapper = SimpleNamespace(
                file=file_obj,
                original_name=file_obj.name,
                modality=modality
            )
            file_wrappers.append(wrapper)

        # FULL STACKING: Stack all modalities into a 4-channel NIfTI
        if extension == '.png':
            from .stacking import stack_png_files
            stacked_volume = stack_png_files(file_wrappers)
            stacked_name = f'stacked_preview_{uuid4().hex}.png'
        else:
            from .stacking import stack_nifti_files
            stacked_volume = stack_nifti_files(file_wrappers)
            stacked_name = f'stacked_preview_{uuid4().hex}.nii.gz'

        # Upload stacked NIfTI to storage
        stacked_key = f'user_{user_id}/stack_preview/{stacked_name}'
        with tempfile.NamedTemporaryFile(suffix='.nii.gz' if extension != '.png' else '.png', delete=False) as tmp:
            stacked_temp_path = tmp.name
            temp_paths.append(stacked_temp_path)
        
        if extension == '.png':
            stacked_volume.save(stacked_temp_path, format='PNG')
        else:
            nib.save(stacked_volume, stacked_temp_path)

        stacked_url = storage.upload(stacked_temp_path, stacked_key)
        if stacked_url.startswith('/media/'):
            stacked_url = _build_public_url(request, stacked_url)

        # Generate quick preview PNG for UI feedback
        preview_source = _pick_preview_source(files, modalities)
        preview_bytes = _build_preview_png_bytes(preview_source)
        preview_b64 = base64.b64encode(preview_bytes).decode('ascii')

        return Response(
            {
                'success': True,
                'status': 'stacked',
                'stacked_url': stacked_url,
                'preview': preview_b64,
                'preview_url': f'data:image/png;base64,{preview_b64}',
                'filename': stacked_name,
                'mode': 'stacked-full-volume',
            },
            status=status.HTTP_200_OK,
        )

    except ValueError as exc:
        return Response(
            {'success': False, 'error': str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return Response(
            {'success': False, 'error': f'Stack error: {str(exc)}'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        for temp_path in temp_paths:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _pick_preview_upload(files, modalities):
    if len(files) == 1:
        return files[0], 'single'

    modality_to_file = {}
    for index, uploaded_file in enumerate(files):
        modality = modalities[index] if index < len(modalities) else None
        if modality:
            modality_to_file[modality] = uploaded_file

    for modality in EXPECTED_MODALITIES:
        if modality in modality_to_file:
            return modality_to_file[modality], modality

    return files[0], 'first'


def _pick_preview_source(files, modalities):
    if len(files) == 1:
        return files[0]

    modality_to_file = {}
    for index, uploaded_file in enumerate(files):
        modality = modalities[index] if index < len(modalities) else None
        if modality:
            modality_to_file[modality] = uploaded_file

    for modality in EXPECTED_MODALITIES:
        if modality in modality_to_file:
            return modality_to_file[modality]

    return files[0]


def _build_preview_png_bytes(uploaded_file):
    extension = infer_extension(uploaded_file.name)

    if extension == '.png':
        from PIL import Image

        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert('L')
    else:
        if hasattr(uploaded_file, 'temporary_file_path'):
            nii = nib.load(uploaded_file.temporary_file_path())
        else:
            uploaded_file.seek(0)
            nii = nib.load(uploaded_file)

        data = np.asarray(nii.dataobj, dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError('NIfTI preview requires a 3D or 4D volume.')

        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]
        s_min, s_max = float(np.min(slice_data)), float(np.max(slice_data))
        if s_max - s_min > 1e-8:
            slice_data = ((slice_data - s_min) / (s_max - s_min) * 255).astype(np.uint8)
        else:
            slice_data = np.zeros_like(slice_data, dtype=np.uint8)

        from PIL import Image
        img = Image.fromarray(slice_data, mode='L')

    max_dim = 320
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.Resampling.LANCZOS)

    output = io.BytesIO()
    img.save(output, format='PNG')
    return output.getvalue()


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


@api_view(['GET'])
def get_segmentation_status(request, job_id):
    """
    GET /api/segment/{id}/status/
    Get the current status and progress of a segmentation job.
    """
    job = get_object_or_404(SegmentationJob, id=job_id)
    serializer = SegmentationJobStatusSerializer(job, context={'request': request})
    data = serializer.data

    # Add error info if applicable
    if job.status in ('error', 'failed'):
        data['error'] = job.error_message

    return Response(data)


@api_view(['GET'])
def get_segmentation_result(request, job_id):
    """
    GET /api/segment/{id}/result/
    Get the full result of a completed segmentation job.
    """
    job = get_object_or_404(SegmentationJob, id=job_id)

    if job.status not in ('done', 'error', 'failed'):
        return Response(
            {
                'error': 'Job is still processing.',
                'status': job.status,
            },
            status=status.HTTP_202_ACCEPTED,
        )

    serializer = SegmentationJobResultSerializer(job, context={'request': request})
    return Response(serializer.data)


@api_view(['GET'])
def download_segmentation(request, job_id):
    """
    GET /api/segment/{id}/download/
    Download the segmentation result file.
    """
    job = get_object_or_404(SegmentationJob, id=job_id)

    if job.status != 'done':
        return Response(
            {'error': 'Segmentation not complete.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if job.segmentation_file:
        from django.http import FileResponse
        return FileResponse(
            job.segmentation_file.open('rb'),
            as_attachment=True,
            filename=f'segmentation_{job.id}.nii.gz',
        )

    if not job.mask_url:
        return Response(
            {'error': 'No segmentation file available.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    # Fallback: stream from storage URL (Supabase/local public URL) when no local FileField exists.
    parsed = urlparse(job.mask_url)
    if not parsed.scheme:
        local_path = Path(settings.MEDIA_ROOT) / job.mask_url.replace('/media/', '')
        if not local_path.exists():
            return Response(
                {'error': 'Segmentation mask file not found.'},
                status=status.HTTP_404_NOT_FOUND,
            )
        from django.http import FileResponse
        return FileResponse(
            open(local_path, 'rb'),
            as_attachment=True,
            filename=f'segmentation_{job.id}.nii.gz',
        )

    try:
        import requests
        from django.http import HttpResponse

        upstream = requests.get(job.mask_url, timeout=120)
        upstream.raise_for_status()

        response = HttpResponse(upstream.content, content_type='application/gzip')
        response['Content-Disposition'] = f'attachment; filename="segmentation_{job.id}.nii.gz"'
        return response
    except Exception as exc:
        return Response(
            {'error': f'Failed to fetch segmentation from storage: {exc}'},
            status=status.HTTP_502_BAD_GATEWAY,
        )
