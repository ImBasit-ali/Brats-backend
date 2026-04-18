import os
import tempfile
import threading
import sys

import requests

MODEL = None
MODEL_LOCK = threading.Lock()
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    '/tmp/model.keras' if os.name != 'nt' else os.path.join(tempfile.gettempdir(), 'model.keras'),
)


def download_model():
    if os.path.exists(MODEL_PATH):
        return

    print("Downloading model from Google Drive...")

    url = os.environ.get("MODEL_URL")
    if not url:
        raise Exception("MODEL_URL environment variable is not set")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    response = requests.get(url, stream=True, timeout=120)
    if response.status_code != 200:
        raise Exception("Failed to download model")

    with open(MODEL_PATH, "wb") as file_handle:
        for chunk in response.iter_content(8192):
            if chunk:
                file_handle.write(chunk)

    print("Model downloaded successfully")


def get_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is None:
            download_model()
            print("Loading model...")
            try:
                from tensorflow.keras.models import load_model
            except Exception as exc:
                raise Exception(
                    'TensorFlow is not installed for this Python version. '
                    f'Current Python is {sys.version.split()[0]}. '
                    'Use Python 3.11 or 3.12 for TensorFlow inference.'
                ) from exc
            MODEL = load_model(MODEL_PATH)
            print("Model loaded successfully")

    return MODEL
