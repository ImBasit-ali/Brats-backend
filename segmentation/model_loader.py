import os
import threading
from pathlib import Path
import urllib.request
import tempfile

os.environ.setdefault('KERAS_BACKEND', 'jax')

MODEL = None
MODEL_LOCK = threading.Lock()
KERAS = None
KERAS_OPS = None


def _download_model_from_url(url: str) -> Path:
    """Download model from URL and save to a temporary location."""
    try:
        print(f"Downloading model from {url}...")
        temp_dir = Path(tempfile.gettempdir()) / 'brats_models'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a stable filename based on the URL
        model_filename = 'brats_3d_unet_final.keras'
        model_path = temp_dir / model_filename
        
        # Download the file
        urllib.request.urlretrieve(url, str(model_path))
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        raise Exception(f"Failed to download model from {url}: {str(e)}")


def _resolve_model_path() -> Path:
    # Check for Railway MODEL_URL environment variable first
    model_url = os.environ.get('MODEL_URL')
    if model_url:
        return _download_model_from_url(model_url)

    # Fall back to local MODEL_PATH, defaulting to backend/model_assets model file
    repo_root = Path(__file__).resolve().parent.parent.parent
    configured_path = Path(
        os.environ.get('MODEL_PATH', 'backend/model_assets/brats_3d_unet_final.keras')
    )

    if configured_path.is_absolute():
        return configured_path

    return (repo_root / configured_path).resolve()


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
            model_url = os.environ.get('MODEL_URL')
            
            try:
                model_path = _resolve_model_path()
                
                if not model_path.exists():
                    if model_url:
                        raise Exception(
                            f'Failed to download model from {model_url}. Please check the URL and try again.'
                        )
                    else:
                        raise Exception(
                            f'Model file not found at {model_path}. Default is backend/model_assets/brats_3d_unet_final.keras. Set MODEL_PATH to a valid .keras file or MODEL_URL to a remote model URL.'
                        )
                
                print("Loading model...")
                MODEL = KERAS.models.load_model(
                    str(model_path),
                    custom_objects={'instance_normalization': instance_normalization},
                    compile=False,
                )
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    return MODEL
