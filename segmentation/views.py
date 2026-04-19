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
from .model_loader import get_model
from .tasks import mock_process_segmentation


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def create_segmentation(request):
    """
    POST /api/segment/
    Upload NIfTI files and start a new segmentation job.
    """
    files = request.FILES.getlist('files')
    modalities = request.POST.getlist('modalities')

    if not files:
        return Response(
            {'error': 'No files uploaded. Please upload at least one NIfTI file.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    print('Request received')
    try:
        get_model()
    except Exception as exc:
        print('ERROR:', str(exc))
        return Response({'error': str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Create job
    grade = request.POST.get('grade', 'HGG')
    regions_str = request.POST.get('regions', '{}')
    opacity = int(request.POST.get('opacity', 70))

    try:
        regions = json.loads(regions_str) if isinstance(regions_str, str) else regions_str
    except (json.JSONDecodeError, TypeError):
        regions = {}

    job = SegmentationJob.objects.create(
        grade=grade,
        regions=regions,
        opacity=opacity,
    )

    # Save uploaded files first (raw inputs)
    for i, file in enumerate(files):
        if len(files) == 4:
            modality = modalities[i] if i < len(modalities) else EXPECTED_MODALITIES[i]
        else:
            modality = 'stacked'
        UploadedFile.objects.create(
            job=job,
            file=file,
            original_name=file.name,
            modality=modality,
        )

    uploaded_files = list(job.files.all())

    try:
        upload_mode, extension = validate_upload_combination(uploaded_files)
    except ValueError as exc:
        job.status = 'error'
        job.error_message = str(exc)
        job.save(update_fields=['status', 'error_message', 'updated_at'])
        return Response({'error': str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    if upload_mode == 'modalities-four':
        stacked_name = f'{job.id}_stacked_input{extension}'
        stacked_file = _create_stacked_file(uploaded_files, extension, stacked_name)
        UploadedFile.objects.create(
            job=job,
            file=stacked_file,
            original_name=stacked_name,
            modality='stacked',
        )
    else:
        single_file = uploaded_files[0]
        stacked_name = f'{job.id}_stacked_input{extension}'
        stacked_file = _create_stacked_file(
            [_wrap_uploaded_file(single_file, modality) for modality in EXPECTED_MODALITIES],
            extension,
            stacked_name,
        )
        UploadedFile.objects.create(
            job=job,
            file=stacked_file,
            original_name=stacked_name,
            modality='stacked',
        )

    # Start mock processing
    mock_process_segmentation(str(job.id))

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
        selected_file, selected_modality = _pick_preview_upload(files, modalities)
        extension = infer_extension(selected_file.name)
        if extension not in ('.nii', '.nii.gz', '.png'):
            raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

        preview_name = f'stacked_preview_{uuid4().hex}{extension}'
        storage_path = default_storage.save(f'previews/{preview_name}', selected_file)
        preview_url = _build_public_url(request, default_storage.url(storage_path))

        return Response(
            {
                'success': True,
                'status': 'stacked',
                'preview_url': preview_url,
                'filename': preview_name,
                'mode': f'preview-{selected_modality}',
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


def _create_stacked_file(uploaded_files, extension, output_name):
    if extension in ('.nii', '.nii.gz'):
        stacked_nifti = stack_nifti_files(uploaded_files)
        suffix = '.nii.gz' if extension == '.nii.gz' else '.nii'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_path = tmp.name
        try:
            nib.save(stacked_nifti, temp_path)
            with open(temp_path, 'rb') as handle:
                return ContentFile(handle.read(), name=output_name)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    if extension == '.png':
        stacked_png = stack_png_files(uploaded_files)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name
        try:
            stacked_png.save(temp_path, format='PNG')
            with open(temp_path, 'rb') as handle:
                return ContentFile(handle.read(), name=output_name)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    raise ValueError(f'Unsupported file extension: {extension}')


def _wrap_uploaded_file(uploaded_file, modality):
    return SimpleNamespace(
        original_name=uploaded_file.original_name,
        modality=modality,
        file=uploaded_file.file,
    )


def _wrap_temp_file(temp_path, original_name, modality):
    return SimpleNamespace(
        original_name=original_name,
        modality=modality,
        file=SimpleNamespace(path=temp_path),
    )


def _write_upload_to_temp(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    if uploaded_file.name.lower().endswith('.nii.gz'):
        suffix = '.nii.gz'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        return tmp.name


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
    serializer = SegmentationJobStatusSerializer(job)
    data = serializer.data

    # Add error info if applicable
    if job.status == 'error':
        data['error'] = job.error_message

    return Response(data)


@api_view(['GET'])
def get_segmentation_result(request, job_id):
    """
    GET /api/segment/{id}/result/
    Get the full result of a completed segmentation job.
    """
    job = get_object_or_404(SegmentationJob, id=job_id)

    if job.status not in ('done', 'error'):
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
            {'error': 'No segmentation file available. This is a mock result.'},
            status=status.HTTP_404_NOT_FOUND,
        )

    from django.http import FileResponse
    return FileResponse(
        job.segmentation_file.open('rb'),
        as_attachment=True,
        filename=f'segmentation_{job.id}.nii.gz',
    )
