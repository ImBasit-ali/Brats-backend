import os
import importlib

MODEL = None


def _resolve_model_path():
    """
    Resolve the model file path.

    Priority:
      1. MODEL_KERAS_PATH env var (explicit override)
      2. /app/model/model.keras  (Docker / Railway container path)
      3. ./model/model.keras     (local development fallback)
    """
    env_path = os.environ.get('MODEL_KERAS_PATH')
    if env_path and os.path.exists(env_path):
        return env_path

    # Docker / Railway container path (COPY . . puts model/ at /app/model/)
    docker_path = '/app/model/model.keras'
    if os.path.exists(docker_path):
        return docker_path

    # Local development path — relative to this file's parent's parent
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_path = os.path.join(BASE_DIR, 'model', 'model.keras')
    return local_path


def _resolve_load_model():
    try:
        tf_models = importlib.import_module('tensorflow.keras.models')
        return tf_models.load_model
    except ImportError:
        raise ImportError(
            'TensorFlow/Keras is not installed. '
            'Please install tensorflow and keras to load the model.'
        )


def instance_normalization(x):
    try:
        tf = importlib.import_module('tensorflow')

        axes = tuple(range(1, len(x.shape) - 1))
        mean = tf.reduce_mean(x, axis=axes, keepdims=True)
        variance = tf.math.reduce_variance(x, axis=axes, keepdims=True)
        return (x - mean) / tf.sqrt(variance + 1e-5)
    except Exception:
        keras_ops = importlib.import_module('keras.ops')
        axes = tuple(range(1, len(x.shape) - 1))
        mean = keras_ops.mean(x, axis=axes, keepdims=True)
        variance = keras_ops.var(x, axis=axes, keepdims=True)
        return (x - mean) / keras_ops.sqrt(variance + 1e-5)


def get_model():
    """
    Load the Keras model exactly once and cache it in the global MODEL variable.

    The model is never unloaded — it stays in memory for the lifetime of the
    gunicorn worker process.  With --workers 2, each process loads the model
    independently (~68 MB × 2 = manageable).
    """
    global MODEL

    if MODEL is None:
        model_path = _resolve_model_path()

        print(f'🔄 Loading model from: {model_path}')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'Model file not found at {model_path}. '
                f'Set MODEL_KERAS_PATH env var or place model.keras at {model_path}'
            )

        load_model = _resolve_load_model()
        MODEL = load_model(
            model_path,
            custom_objects={'instance_normalization': instance_normalization},
            compile=False,
        )

        print('✅ Model loaded successfully')

    return MODEL