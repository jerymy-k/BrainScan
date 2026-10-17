"""End-to-end training pipeline: clean -> resize -> load -> split -> train -> evaluate.

Usage:
    brainscan-train                             # full run, same settings as the notebook
    brainscan-train --epochs 1 --model-path /tmp/smoke.keras --reports-dir /tmp/reports

The best epoch (lowest validation loss) is saved to ``--model-path`` and that
checkpoint, not the last epoch, is the one evaluated on the test set.
"""

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from brainscan.config import (
    BATCH_SIZE, DATA_DIR, EPOCHS, MODEL_PATH, REPORTS_DIR, RESIZED_DIR, SEED, VALIDATION_SPLIT,
)
from brainscan.data import clean_dataset, load_dataset, resize_dataset, split_dataset
from brainscan.evaluate import evaluate_model, print_summary, save_training_curves, write_reports
from brainscan.model import build_model


def train(X_train, y_train, model_path, epochs=EPOCHS, batch_size=BATCH_SIZE, early_stopping=False):
    """Train a fresh model, checkpointing the epoch with the lowest validation loss.

    Returns the Keras History object.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)]
    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))

    model = build_model()
    return model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        shuffle=True,
    )


def main():
    parser = argparse.ArgumentParser(description='Train the BrainScan CNN end to end.')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR, help='Raw images, one folder per class.')
    parser.add_argument('--resized-dir', type=Path, default=RESIZED_DIR)
    parser.add_argument('--model-path', type=Path, default=MODEL_PATH)
    parser.add_argument('--reports-dir', type=Path, default=REPORTS_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--remove-bad-files', action='store_true',
                        help='Delete invalid/corrupted files from --data-dir (default: only report them).')
    args = parser.parse_args()

    # Seeds Python, NumPy and TensorFlow. GPU kernels can still add tiny
    # run-to-run differences, so expect very close (not bit-identical) results.
    tf.keras.utils.set_random_seed(args.seed)

    clean_dataset(args.data_dir, remove=args.remove_bad_files)
    resize_dataset(args.data_dir, args.resized_dir)
    images, labels = load_dataset(args.resized_dir)
    X_train, X_test, y_train, y_test = split_dataset(images, labels, seed=args.seed)
    print(f'Train: {X_train.shape}  Test: {X_test.shape}')

    start = time.perf_counter()
    history = train(X_train, y_train, args.model_path, args.epochs, args.batch_size, args.early_stopping)
    minutes = (time.perf_counter() - start) / 60
    print(f'Training time: {minutes:.1f} min')

    best_model = tf.keras.models.load_model(args.model_path)
    results = evaluate_model(best_model, X_test, y_test)
    results['training_minutes'] = round(minutes, 2)
    results['epochs'] = args.epochs
    write_reports(results, args.reports_dir)
    save_training_curves(history.history, Path(args.reports_dir) / 'training_curves.png')
    (Path(args.reports_dir) / 'history.json').write_text(json.dumps(history.history, indent=2))
    print_summary(results)


if __name__ == '__main__':
    main()
