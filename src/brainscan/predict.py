"""Predict the tumour class of a single MRI image with the trained model."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from brainscan.config import CLASS_NAMES, MODEL_PATH, TARGET_SIZE, VALID_EXTENSIONS


class InvalidImageError(ValueError):
    """Raised when a file is not a usable image (bad format, empty or corrupted)."""


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    """Load the trained Keras model once and reuse it on later calls."""
    return tf.keras.models.load_model(model_path)


def preprocess_image(path):
    """Validate an image file and turn it into a model-ready batch of one.

    Applies the same steps as training: OpenCV BGR image resized to 224x224
    with INTER_AREA. Pixels stay in 0-255 because the model rescales them
    itself (Rescaling(1./255) layer).

    Raises:
        FileNotFoundError: the path does not exist or is not a file.
        InvalidImageError: unsupported extension, empty, truncated or corrupted file.
    """
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
    """Classify one MRI image.

    Args:
        path: path to a .jpg/.jpeg/.png/.bmp image.
        model: an already loaded Keras model; defaults to the saved best model.

    Returns:
        dict with ``label`` (predicted class), ``confidence`` (its probability)
        and ``probabilities`` (class name -> probability for all 4 classes).
    """
    if model is None:
        model = load_model()
    probs = model.predict(preprocess_image(path), verbose=0)[0]
    predicted = int(np.argmax(probs))
    return {
        'label': CLASS_NAMES[predicted],
        'confidence': float(probs[predicted]),
        'probabilities': {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
