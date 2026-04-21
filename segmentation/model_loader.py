import os
import threading
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'jax')

MODEL = None
MODEL_LOCK = threading.Lock()
KERAS = None
KERAS_OPS = None


def _resolve_model_path() -> Path:
    configured_path = Path(
        os.environ.get('MODEL_PATH', 'model_assets/brats_3d_unet_final.keras')
    )

    if configured_path.is_absolute():
        return configured_path

    backend_root = Path(__file__).resolve().parent.parent
    return (backend_root / configured_path).resolve()


def instance_normalization(x):
    if KERAS_OPS is None:
        raise RuntimeError('Keras backend ops are not initialized.')
    axes = tuple(range(1, len(x.shape) - 1))
    mean = KERAS_OPS.mean(x, axis=axes, keepdims=True)
    variance = KERAS_OPS.var(x, axis=axes, keepdims=True)
    return (x - mean) / KERAS_OPS.sqrt(variance + 1e-5)


def _ensure_keras_loaded():
    global KERAS
    global KERAS_OPS

    if KERAS is not None and KERAS_OPS is not None:
        return

    import keras
    from keras import ops

    KERAS = keras
    KERAS_OPS = ops


def get_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is None:
            _ensure_keras_loaded()
            model_path = _resolve_model_path()
            if not model_path.exists():
                raise Exception(
                    f'Model file not found at {model_path}. Set MODEL_PATH to a valid .keras file.'
                )
            print("Loading model...")
            MODEL = KERAS.models.load_model(
                str(model_path),
                custom_objects={'instance_normalization': instance_normalization},
                compile=False,
            )
            print("Model loaded successfully")

    return MODEL
