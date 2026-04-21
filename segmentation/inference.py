import nibabel as nib
import numpy as np

from .model_loader import get_model


THRESHOLD = 0.5


def run_nifti_model_inference(stacked_path):
    """Run model inference on a stacked NIfTI volume and return ET/WT/TC masks."""
    nii = nib.load(stacked_path)
    volume = np.asarray(nii.dataobj, dtype=np.float32)

    if volume.ndim == 3:
        volume = volume[..., np.newaxis]
    if volume.ndim != 4:
        raise ValueError('Stacked NIfTI must be 4D (x, y, z, channels).')

    model = get_model()
    prepared, mapping, channels_first = _prepare_for_model(volume, model)

    prediction = model.predict(prepared, verbose=0)
    prediction = _unwrap_prediction(prediction)
    prediction = _to_channels_last(prediction, channels_first)

    et_pred, wt_pred, tc_pred = _split_prediction_channels(prediction)

    original_shape = volume.shape[:3]
    et = _restore_to_original_shape(et_pred, mapping, original_shape)
    wt = _restore_to_original_shape(wt_pred, mapping, original_shape)
    tc = _restore_to_original_shape(tc_pred, mapping, original_shape)

    # Limit overlays to anatomical foreground to avoid slab/box artifacts.
    brain_mask = _build_brain_mask(volume[..., 0])
    et = np.logical_and(et, brain_mask)
    wt = np.logical_and(wt, brain_mask)
    tc = np.logical_and(tc, brain_mask)

    return et.astype(np.uint8), wt.astype(np.uint8), tc.astype(np.uint8), nii.affine, nii.header.copy()


def _build_brain_mask(channel):
    channel = np.asarray(channel, dtype=np.float32)
    positive = channel[channel > 0]
    if positive.size == 0:
        return np.zeros_like(channel, dtype=bool)

    # Use a small foreground threshold to retain brain tissue and suppress empty background.
    threshold = float(np.percentile(positive, 2.0))
    return channel > threshold


def _prepare_for_model(volume, model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if len(input_shape) != 5:
        raise ValueError(f'Expected a 3D model input shape of rank 5, got: {input_shape}')

    channels_first = _is_channels_first(input_shape)

    if channels_first:
        target_spatial = _resolve_target_shape(input_shape[2:5], volume.shape[:3])
        target_channels = input_shape[1] if input_shape[1] is not None else volume.shape[-1]
    else:
        target_spatial = _resolve_target_shape(input_shape[1:4], volume.shape[:3])
        target_channels = input_shape[4] if input_shape[4] is not None else volume.shape[-1]

    prepared = _normalize_per_channel(volume)
    prepared, mapping = _center_crop_or_pad(prepared, target_spatial)
    prepared = _align_channels(prepared, int(target_channels))

    if channels_first:
        prepared = np.transpose(prepared, (3, 0, 1, 2))

    prepared = np.expand_dims(prepared, axis=0)
    return prepared.astype(np.float32), mapping, channels_first


def _unwrap_prediction(prediction):
    if isinstance(prediction, list):
        if not prediction:
            raise ValueError('Model returned an empty prediction list.')
        prediction = prediction[0]

    prediction = np.asarray(prediction)
    if prediction.ndim == 5 and prediction.shape[0] == 1:
        prediction = prediction[0]

    if prediction.ndim not in (3, 4):
        raise ValueError(f'Unexpected prediction shape: {prediction.shape}')

    if prediction.ndim == 3:
        prediction = prediction[..., np.newaxis]

    return prediction


def _to_channels_last(prediction, channels_first):
    if channels_first:
        return np.transpose(prediction, (1, 2, 3, 0))
    return prediction


def _split_prediction_channels(prediction):
    channels = prediction.shape[-1]

    if channels >= 3:
        et = prediction[..., 0] > THRESHOLD
        wt = prediction[..., 1] > THRESHOLD
        tc = prediction[..., 2] > THRESHOLD
        return et, wt, tc

    if channels == 2:
        et = prediction[..., 0] > THRESHOLD
        wt = prediction[..., 1] > THRESHOLD
        tc = et | wt
        return et, wt, tc

    if channels == 1:
        wt = prediction[..., 0] > THRESHOLD
        et = wt.copy()
        tc = wt.copy()
        return et, wt, tc

    raise ValueError('Prediction output has no channels.')


def _resolve_target_shape(target_shape, fallback_shape):
    return tuple(int(ts if ts is not None else fallback_shape[i]) for i, ts in enumerate(target_shape))


def _is_channels_first(input_shape):
    # Heuristic: 3D models are usually channels-last, but some exports are channels-first.
    channels_dim = input_shape[1]
    if channels_dim in (1, 2, 3, 4):
        return True

    last_dim = input_shape[-1]
    if last_dim in (1, 2, 3, 4):
        return False

    return False


def _normalize_per_channel(volume):
    normalized = np.zeros_like(volume, dtype=np.float32)
    for idx in range(volume.shape[-1]):
        channel = volume[..., idx]
        c_min = float(np.min(channel))
        c_max = float(np.max(channel))
        if c_max - c_min < 1e-8:
            continue
        normalized[..., idx] = (channel - c_min) / (c_max - c_min)
    return normalized


def _center_crop_or_pad(volume, target_spatial):
    src_shape = volume.shape[:3]
    channels = volume.shape[-1]
    output = np.zeros((*target_spatial, channels), dtype=volume.dtype)

    src_slices = []
    dst_slices = []

    for src, dst in zip(src_shape, target_spatial):
        if src >= dst:
            src_start = (src - dst) // 2
            src_end = src_start + dst
            dst_start = 0
            dst_end = dst
        else:
            src_start = 0
            src_end = src
            dst_start = (dst - src) // 2
            dst_end = dst_start + src

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    output[
        dst_slices[0], dst_slices[1], dst_slices[2], :
    ] = volume[
        src_slices[0], src_slices[1], src_slices[2], :
    ]

    return output, {'src': tuple(src_slices), 'dst': tuple(dst_slices), 'target_shape': target_spatial}


def _align_channels(volume, target_channels):
    current_channels = volume.shape[-1]
    if target_channels == current_channels:
        return volume

    if target_channels < current_channels:
        return volume[..., :target_channels]

    pad = np.zeros((*volume.shape[:3], target_channels - current_channels), dtype=volume.dtype)
    return np.concatenate([volume, pad], axis=-1)


def _restore_to_original_shape(mask, mapping, original_shape):
    restored = np.zeros(original_shape, dtype=np.uint8)
    src = mapping['src']
    dst = mapping['dst']

    restored[src[0], src[1], src[2]] = mask[dst[0], dst[1], dst[2]].astype(np.uint8)
    return restored
