"""Dataset cleaning, resizing, loading and train/test splitting.

Mirrors the steps of notebooks/main.ipynb so the whole pipeline can run from
the command line (see brainscan.train).
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from brainscan.config import CLASS_NAMES, SEED, TARGET_SIZE, TEST_SIZE, VALID_EXTENSIONS


def find_bad_files(data_dir):
    """Return files that cannot be used for training.

    A file is bad if its extension is not an image extension, if it is empty,
    or if OpenCV cannot decode it.
    """
    bad_files = []
    for file in sorted(Path(data_dir).rglob('*')):
        if not file.is_file():
            continue
        if (
            file.suffix.lower() not in VALID_EXTENSIONS
            or file.stat().st_size == 0
            or cv2.imread(str(file)) is None
        ):
            bad_files.append(file)
    return bad_files


def clean_dataset(data_dir, remove=False):
    """Report bad files in ``data_dir`` and delete them when ``remove`` is True."""
    bad_files = find_bad_files(data_dir)
    print(f'Bad files found: {len(bad_files)}')
    for file in bad_files:
        print(f' -> {file}')
        if remove:
            file.unlink()
    return bad_files


def resize_dataset(data_dir, output_dir, size=TARGET_SIZE):
    """Write a resized copy of every image, keeping the ``<class>/<file>`` layout.

    Images already present in ``output_dir`` are skipped, so this is cheap to rerun.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    written = 0
    for file in sorted(data_dir.rglob('*')):
        if not file.is_file() or file.suffix.lower() not in VALID_EXTENSIONS:
            continue
        save_path = output_dir / file.relative_to(data_dir)
        if save_path.exists():
            continue
        image = cv2.imread(str(file))
        if image is None:
            print(f'Skipping unreadable image: {file}')
            continue
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        written += 1
    print(f'Resized images written: {written}')


def load_dataset(image_dir):
    """Load all images (BGR, as read by OpenCV) and their class labels.

    The label is the name of the image's parent folder. Files are read in
    ``rglob`` order, the same order as the notebook, so the split below gives
    the exact same train/test sets.
    """
    images, labels = [], []
    for file in Path(image_dir).rglob('*'):
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
            images.append(cv2.imread(str(file)))
            labels.append(file.parent.name)
    return np.array(images), np.array(labels)


def split_dataset(images, labels, test_size=TEST_SIZE, seed=SEED):
    """Stratified train/test split, with labels encoded as integers.

    Returns ``X_train, X_test, y_train, y_test`` where the ``y`` arrays hold
    indices into ``CLASS_NAMES``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=seed, shuffle=True, stratify=labels
    )
    encoder = LabelEncoder().fit(y_train)
    assert list(encoder.classes_) == CLASS_NAMES, f'Unexpected classes: {encoder.classes_}'
    return X_train, X_test, encoder.transform(y_train), encoder.transform(y_test)
