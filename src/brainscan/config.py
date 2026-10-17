"""Project-wide constants: paths, image size, classes and training defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'Data'
RESIZED_DIR = PROJECT_ROOT / 'Resized_data'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
MODEL_PATH = MODELS_DIR / 'best_brain_tumor_model.keras'

TARGET_SIZE = (224, 224)
VALID_EXTENSIONS = {'.jpg', '.png', '.jpeg', '.bmp'}
# Alphabetical, i.e. the order LabelEncoder assigns: index i <-> CLASS_NAMES[i]
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

SEED = 42
TEST_SIZE = 0.15
VALIDATION_SPLIT = 0.15
EPOCHS = 30
BATCH_SIZE = 32
