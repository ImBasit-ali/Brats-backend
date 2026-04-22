import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import nibabel as nib
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
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
    stack_nifti_files,
    stack_png_files,
    validate_upload_combination,
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

    # Save uploaded files
    input_keys = []
    for i, file in enumerate(files):
        if len(files) == 4:
            modality = modalities[i] if i < len(modalities) else EXPECTED_MODALITIES[i]
        else:
            modality = 'stacked'
        uploaded_record = UploadedFile.objects.create(
            job=job,
            file=file,
            original_name=file.name,
            modality=modality,
        )
        input_keys.append(uploaded_record.file.name)

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
    Stack the uploaded modalities without starting segmentation.
    """
    files = request.FILES.getlist('files')
    modalities = request.POST.getlist('modalities')

    if not files:
        named_files = [
            ('t1', request.FILES.get('t1')),
            ('t1ce', request.FILES.get('t1ce')),
            ('t2', request.FILES.get('t2')),
            ('flair', request.FILES.get('flair')),
        ]
        files = [file_obj for _, file_obj in named_files if file_obj is not None]
        modalities = [modality for modality, file_obj in named_files if file_obj is not None]

    if not files:
        return Response(
            {'success': False, 'error': 'No files uploaded. Please upload at least one NIfTI file.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        # Create UploadedFile objects for stacking
        uploaded_file_objs = []
        for i, file_obj in enumerate(files):
            modality = modalities[i] if i < len(modalities) else 't1'
            # Create a temporary UploadedFile object for stacking
            temp_uploaded = SimpleNamespace(
                file=SimpleNamespace(path=None),
                original_name=file_obj.name,
                modality=modality
            )
            uploaded_file_objs.append(temp_uploaded)

        # Determine extension from first file
        extension = infer_extension(files[0].name)
        if extension not in ('.nii', '.nii.gz', '.png'):
            raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

        # Stack files if there are 4 modalities, otherwise use the single file
        if len(files) == 4:
            # Save uploaded files to temporary location for stacking
            temp_paths = []
            for file_obj in files:
                temp_path = default_storage.save(f'temp/{uuid4().hex}_{file_obj.name}', file_obj)
                temp_paths.append(temp_path)

            try:
                # Load the files and stack them
                temp_file_objs = []
                for temp_path, original_name, modality in zip(temp_paths, [f.name for f in files], modalities):
                    full_path = default_storage.path(temp_path)
                    temp_obj = SimpleNamespace(
                        file=SimpleNamespace(path=full_path),
                        original_name=original_name,
                        modality=modality
                    )
                    temp_file_objs.append(temp_obj)

                if extension == '.png':
                    stacked_image = stack_png_files(temp_file_objs)
                    # Save stacked PNG
                    stacked_name = f'stacked_preview_{uuid4().hex}.png'
                    img_bytes = io.BytesIO()
                    stacked_image.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    storage_path = default_storage.save(f'previews/{stacked_name}', ContentFile(img_bytes.read()))
                    mode = 'stacked-4-modalities'
                else:
                    # Stack NIfTI files
                    stacked_nii = stack_nifti_files(temp_file_objs)
                    # Save stacked NIfTI
                    stacked_name = f'stacked_preview_{uuid4().hex}.nii.gz'
                    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
                        nib.save(stacked_nii, tmp.name)
                        tmp_path = tmp.name

                    with open(tmp_path, 'rb') as f:
                        storage_path = default_storage.save(f'previews/{stacked_name}', ContentFile(f.read()))
                    
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                    mode = 'stacked-4-modalities'

                preview_url = _build_public_url(request, default_storage.url(storage_path))

                # Clean up temporary uploaded files
                for temp_path in temp_paths:
                    try:
                        default_storage.delete(temp_path)
                    except:
                        pass

                return Response(
                    {
                        'success': True,
                        'status': 'stacked',
                        'preview_url': preview_url,
                        'filename': stacked_name,
                        'mode': mode,
                    },
                    status=status.HTTP_200_OK,
                )
            finally:
                # Clean up temporary files if something went wrong
                for temp_path in temp_paths:
                    try:
                        default_storage.delete(temp_path)
                    except:
                        pass
        else:
            # Single file - just save and return
            preview_name = f'stacked_preview_{uuid4().hex}{extension}'
            storage_path = default_storage.save(f'previews/{preview_name}', files[0])
            preview_url = _build_public_url(request, default_storage.url(storage_path))

            return Response(
                {
                    'success': True,
                    'status': 'stacked',
                    'preview_url': preview_url,
                    'filename': preview_name,
                    'mode': f'preview-single',
                },
                status=status.HTTP_200_OK,
            )

    except ValueError as exc:
        return Response(
            {'success': False, 'error': str(exc)},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as exc:
        return Response(
            {'success': False, 'error': str(exc)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


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

    if not job.segmentation_file:
        return Response(
            {'error': 'No segmentation file available.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    from django.http import FileResponse
    return FileResponse(
        job.segmentation_file.open('rb'),
        as_attachment=True,
        filename=f'segmentation_{job.id}.nii.gz',
    )
