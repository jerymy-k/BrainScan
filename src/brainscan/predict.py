from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'best_brain_tumor_model.keras'
TARGET_SIZE = (224, 224)
# Same order as LabelEncoder().classes_ in the training notebook (alphabetical)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    return tf.keras.models.load_model(model_path)


def preprocess_image(path):
    # Same steps as training: cv2 BGR image, resized to 224x224 with INTER_AREA.
    # No /255 here: the model has its own Rescaling(1./255) layer.
    image = cv2.imread(str(path))
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
