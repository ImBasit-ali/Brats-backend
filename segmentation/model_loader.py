import os
import importlib

MODEL = None


def _resolve_load_model():
    try:
        tf_models = importlib.import_module("tensorflow.keras.models")
        return tf_models.load_model
    except ImportError:
        raise ImportError("TensorFlow/Keras is not installed. Please install tensorflow and keras to load the model.")


def instance_normalization(x):
    try:
        tf = importlib.import_module("tensorflow")

        axes = tuple(range(1, len(x.shape) - 1))
        mean = tf.reduce_mean(x, axis=axes, keepdims=True)
        variance = tf.math.reduce_variance(x, axis=axes, keepdims=True)
        return (x - mean) / tf.sqrt(variance + 1e-5)
    except Exception:
        keras_ops = importlib.import_module("keras.ops")
        axes = tuple(range(1, len(x.shape) - 1))
        mean = keras_ops.mean(x, axis=axes, keepdims=True)
        variance = keras_ops.var(x, axis=axes, keepdims=True)
        return (x - mean) / keras_ops.sqrt(variance + 1e-5)

def get_model():
    global MODEL

    if MODEL is None:
        try:
            print("🔄 Loading model...")

            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(BASE_DIR, "model", "model.keras")

            print("📂 Model path:", model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            load_model = _resolve_load_model()
            MODEL = load_model(
                model_path,
                custom_objects={'instance_normalization': instance_normalization},
                compile=False,
            )

            print("✅ Model loaded successfully")

        except Exception as e:
            print("❌ MODEL LOAD ERROR:", str(e))
            raise e  # VERY IMPORTANT → don't hide error

    return MODEL