from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'best_brain_tumor_model.keras'
TARGET_SIZE = (224, 224)
# Same order as LabelEncoder().classes_ in the training notebook (alphabetical)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
VALID_EXTENSIONS = {'.jpg', '.png', '.jpeg', '.bmp'}


class InvalidImageError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def preprocess_image(path):
    # Same steps as training: cv2 BGR image, resized to 224x224 with INTER_AREA.
    # No /255 here: the model has its own Rescaling(1./255) layer.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in VALID_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported format '{path.suffix}' (expected one of {sorted(VALID_EXTENSIONS)})"
        )
    if path.stat().st_size == 0:
        raise InvalidImageError(f"Empty file: {path}")
    try:
        # Full decode: catches truncated files that cv2.imread silently fills with grey
        with Image.open(path) as img:
            img.load()
    except (OSError, SyntaxError) as e:
        raise InvalidImageError(f"Corrupted or unreadable image: {path}") from e

    image = cv2.imread(str(path))
    if image is None:
        raise InvalidImageError(f"Corrupted or unreadable image: {path}")
    image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return np.expand_dims(image, axis=0)


def predict_image(path, model=None):
    if model is None:
        model = load_model()
    probs = model.predict(preprocess_image(path), verbose=0)[0]
    predicted = int(np.argmax(probs))
    return {
        'label': CLASS_NAMES[predicted],
        'confidence': float(probs[predicted]),
        'probabilities': {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
