from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from .models import UploadedFile

ALLOWED_EXTENSIONS = ('.nii', '.nii.gz', '.png')
EXPECTED_MODALITIES = ('t1', 't1ce', 't2', 'flair')


def infer_extension(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith('.nii.gz'):
        return '.nii.gz'
    if lower.endswith('.nii'):
        return '.nii'
    if lower.endswith('.png'):
        return '.png'
    return ''


def _missing_modalities(modalities):
    return [modality for modality in EXPECTED_MODALITIES if modality not in modalities]


def _duplicate_modalities(modalities):
    duplicates = []
    for modality in EXPECTED_MODALITIES:
        if modalities.count(modality) > 1:
            duplicates.append(modality)
    return duplicates


def validate_upload_combination(uploaded_files: List[UploadedFile]) -> Tuple[str, str]:
    """
    Validate user input shape and return (mode, extension).

    mode:
    - stacked-single: one pre-stacked file
    - modalities-four: four modality-specific files to be stacked here
    """
    extensions = [infer_extension(item.original_name) for item in uploaded_files]
    if any(ext not in ALLOWED_EXTENSIONS for ext in extensions):
        raise ValueError('Only .nii, .nii.gz, and .png files are supported.')

    if len(set(extensions)) != 1:
        raise ValueError('All uploaded files must use the same format (.nii/.nii.gz or .png).')

    extension = extensions[0]
    count = len(uploaded_files)
    if count == 1:
        return 'stacked-single', extension

    modalities = [item.modality for item in uploaded_files]

    if count in (2, 3):
        duplicates = _duplicate_modalities(modalities)
        if duplicates:
            raise ValueError(f'File already uploaded for modality {duplicates[0].upper()}. Each modality can only be uploaded once.')

        missing = _missing_modalities(modalities)
        raise ValueError(
            f'Four modalities are required to stack. Upload {", ".join(m.upper() for m in missing)} or duplicate one of the existing files to fill the missing slot.'
        )

    if count != 4:
        raise ValueError('Upload either exactly 4 modality files or exactly 1 file to duplicate into a stacked input.')

    duplicates = _duplicate_modalities(modalities)
    if duplicates:
        raise ValueError(f'File already uploaded for modality {duplicates[0].upper()}. Each modality can only be uploaded once.')

    if set(modalities) != set(EXPECTED_MODALITIES):
        missing = _missing_modalities(modalities)
        raise ValueError(
            f'When uploading 4 files, modalities must include t1, t1ce, t2, and flair exactly once. Missing: {", ".join(m.upper() for m in missing)}.'
        )

    return 'modalities-four', extension


def stack_nifti_files(uploaded_files: List[UploadedFile]) -> nib.Nifti1Image:
    ordered = sorted(uploaded_files, key=lambda f: EXPECTED_MODALITIES.index(f.modality))

    volumes = []
    reference_shape = None
    reference_affine = None
    reference_header = None

    for item in ordered:
        nii = nib.load(item.file.path)
        data = np.asarray(nii.dataobj, dtype=np.float32)

        if data.ndim == 4:
            if data.shape[-1] == 1:
                data = data[..., 0]
            else:
                raise ValueError(f'{item.original_name} is already multi-channel; upload it as a single stacked file instead.')

        if data.ndim != 3:
            raise ValueError(f'{item.original_name} must be a 3D NIfTI volume.')

        if reference_shape is None:
            reference_shape = data.shape
            reference_affine = nii.affine
            reference_header = nii.header.copy()
        elif data.shape != reference_shape:
            raise ValueError('All modality NIfTI files must have identical dimensions.')

        volumes.append(data)

    stacked = np.stack(volumes, axis=-1)
    return nib.Nifti1Image(stacked.astype(np.float32), reference_affine, reference_header)


def stack_png_files(uploaded_files: List[UploadedFile]) -> Image.Image:
    ordered = sorted(uploaded_files, key=lambda f: EXPECTED_MODALITIES.index(f.modality))

    channels = []
    width = height = None

    for item in ordered:
        img = Image.open(item.file.path).convert('L')
        if width is None:
            width, height = img.size
        elif img.size != (width, height):
            raise ValueError('All PNG modality files must have identical image dimensions.')
        channels.append(img)

    return Image.merge('RGBA', channels)
