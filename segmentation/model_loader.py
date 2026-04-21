import os
import importlib


def _resolve_load_model():
    try:
        tf_models = importlib.import_module("tensorflow.keras.models")
        return tf_models.load_model
    except Exception:
        keras_models = importlib.import_module("keras.models")
        return keras_models.load_model

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        print("Loading model...")
        model_path = os.path.join(os.getcwd(), "model", "model.keras")
        load_model = _resolve_load_model()
        MODEL = load_model(model_path)
        print("Model loaded successfully")
        try:
          model = get_model()
        except Exception as e:
          print("❌ MODEL ERROR:", str(e))
        
    return MODEL
