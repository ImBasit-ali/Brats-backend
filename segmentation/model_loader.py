import os
import importlib

MODEL = None


def _resolve_load_model():
    try:
        tf_models = importlib.import_module("tensorflow.keras.models")
        return tf_models.load_model
    except Exception:
        keras_models = importlib.import_module("keras.models")
        return keras_models.load_model


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
            MODEL = load_model(model_path)

            print("✅ Model loaded successfully")

        except Exception as e:
            print("❌ MODEL LOAD ERROR:", str(e))
            raise e  # VERY IMPORTANT → don't hide error

    return MODEL