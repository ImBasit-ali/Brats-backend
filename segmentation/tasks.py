"""
Mock async processing for segmentation jobs.
Simulates the segmentation pipeline with delays.
In production, replace with actual Celery tasks and U-Net inference.
"""

import threading
import time
import random
import os
import tempfile

import nibabel as nib
import numpy as np
from django.core.files.base import ContentFile
from django.utils import timezone
from PIL import Image

from .inference import run_nifti_model_inference


def mock_process_segmentation(job_id):
    """
    Simulate the segmentation pipeline in a background thread.
    Steps: Preprocess → Inference → Postprocess → Done
    """
    thread = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    thread.start()


def _run_pipeline(job_id):
    """Run the mock pipeline."""
    from .models import SegmentationJob

    steps = [
        ('Preprocess', 3),
        ('Inference', 5),
        ('Postprocess', 3),
        ('Done', 0),
    ]

    try:
        job = SegmentationJob.objects.get(id=job_id)
        job.status = 'processing'
        job.save()

        for i, (step_name, duration) in enumerate(steps):
            job.current_step = i + 1
            job.current_step_name = step_name
            job.save()

            if duration > 0:
                # Add some randomness to simulate real processing
                time.sleep(duration + random.uniform(-0.5, 1.0))

        # Generate mock outputs + metrics
        _generate_mock_outputs(job)
        job.metrics = _generate_mock_metrics()
        job.status = 'done'
        job.completed_at = timezone.now()
        job.save()

    except SegmentationJob.DoesNotExist:
        pass
    except Exception as e:
        try:
            job = SegmentationJob.objects.get(id=job_id)
            job.status = 'error'
            job.error_message = str(e)
            job.save()
        except SegmentationJob.DoesNotExist:
            pass


def _generate_mock_metrics():
    """Generate realistic mock segmentation metrics."""
    def rand_metric(base, spread=0.05):
        return round(max(0, min(1, base + random.uniform(-spread, spread))), 3)

    return {
        'ET': {
            'volume_ml': round(random.uniform(5, 25), 1),
            'dsc': rand_metric(0.87),
            'sensitivity': rand_metric(0.91),
            'specificity': rand_metric(0.98),
        },
        'NETC': {
            'volume_ml': round(random.uniform(3, 15), 1),
            'dsc': rand_metric(0.82),
            'sensitivity': rand_metric(0.85),
            'specificity': rand_metric(0.97),
        },
        'SNFH': {
            'volume_ml': round(random.uniform(20, 70), 1),
            'dsc': rand_metric(0.90),
            'sensitivity': rand_metric(0.93),
            'specificity': rand_metric(0.99),
        },
        'RC': {
            'volume_ml': round(random.uniform(1, 10), 1),
            'dsc': rand_metric(0.78),
            'sensitivity': rand_metric(0.80),
            'specificity': rand_metric(0.96),
        },
    }


def _generate_mock_outputs(job):
    """Create ET/WT/TC mask files based on the stacked model input."""
    from .models import UploadedFile

    stacked = job.files.filter(modality='stacked').order_by('-uploaded_at').first()
    if not stacked:
        raise ValueError('No stacked model input file found for this job.')

    name_lower = stacked.original_name.lower()
    if name_lower.endswith('.nii') or name_lower.endswith('.nii.gz'):
        et_mask, wt_mask, tc_mask, affine, header = run_nifti_model_inference(stacked.file.path)
        et_file = _save_nifti_temp(et_mask, affine, header.copy(), 'et_mask')
        wt_file = _save_nifti_temp(wt_mask, affine, header.copy(), 'wt_mask')
        tc_file = _save_nifti_temp(tc_mask, affine, header.copy(), 'tc_mask')
        et_ext = wt_ext = tc_ext = '.nii.gz'
    elif name_lower.endswith('.png'):
        et_file, wt_file, tc_file = _create_png_masks(stacked)
        et_ext = wt_ext = tc_ext = '.png'
    else:
        raise ValueError('Unsupported stacked input format for output generation.')

    et_record = UploadedFile.objects.create(
        job=job,
        file=et_file,
        original_name=f'{job.id}_et_mask{et_ext}',
        modality='et_mask',
    )
    wt_record = UploadedFile.objects.create(
        job=job,
        file=wt_file,
        original_name=f'{job.id}_wt_mask{wt_ext}',
        modality='wt_mask',
    )
    UploadedFile.objects.create(
        job=job,
        file=tc_file,
        original_name=f'{job.id}_tc_mask{tc_ext}',
        modality='tc_mask',
    )

    # Use whole tumor mask as the main downloadable segmentation file.
    job.segmentation_file = wt_record.file


def _create_nifti_masks(stacked_file):
    nii = nib.load(stacked_file.file.path)
    data = nii.get_fdata(dtype=np.float32)

    base = data[..., 0] if data.ndim == 4 else data
    base = _normalize(base)

    et = (base > 0.72).astype(np.uint8)
    wt = (base > 0.55).astype(np.uint8)
    tc = ((base > 0.62) & (base <= 0.82)).astype(np.uint8)

    return (
        _save_nifti_temp(et, nii.affine, nii.header.copy(), 'et_mask'),
        _save_nifti_temp(wt, nii.affine, nii.header.copy(), 'wt_mask'),
        _save_nifti_temp(tc, nii.affine, nii.header.copy(), 'tc_mask'),
    )


def _create_png_masks(stacked_file):
    img = Image.open(stacked_file.file.path).convert('RGBA')
    arr = np.asarray(img, dtype=np.float32)

    base = _normalize(arr[..., 0])
    et = (base > 0.72).astype(np.uint8) * 255
    wt = (base > 0.55).astype(np.uint8) * 255
    tc = (((base > 0.62) & (base <= 0.82)).astype(np.uint8)) * 255

    return (
        _save_png_temp(et, 'et_mask'),
        _save_png_temp(wt, 'wt_mask'),
        _save_png_temp(tc, 'tc_mask'),
    )


def _save_nifti_temp(mask_data, affine, header, label):
    header.set_data_dtype(np.uint8)
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        temp_path = tmp.name
    try:
        nib.save(nib.Nifti1Image(mask_data, affine, header), temp_path)
        with open(temp_path, 'rb') as handle:
            return ContentFile(handle.read(), name=f'{label}.nii.gz')
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _save_png_temp(mask_data, label):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        temp_path = tmp.name
    try:
        Image.fromarray(mask_data, mode='L').save(temp_path, format='PNG')
        with open(temp_path, 'rb') as handle:
            return ContentFile(handle.read(), name=f'{label}.png')
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _normalize(arr):
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min)
