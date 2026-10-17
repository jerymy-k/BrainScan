"""Evaluate a saved model on the held-out test set and write report files.

Usage:
    brainscan-evaluate                      # evaluates models/best_brain_tumor_model.keras
    brainscan-evaluate --model other.keras --reports-dir reports/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from brainscan.config import CLASS_NAMES, MODEL_PATH, REPORTS_DIR, RESIZED_DIR, SEED
from brainscan.data import load_dataset, split_dataset


def evaluate_model(model, X_test, y_test):
    """Return test loss/accuracy, the per-class report and the confusion matrix."""
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    return {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy),
        'n_test': int(len(y_test)),
        'classification_report': classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }


def save_confusion_matrix(matrix, path):
    """Save the confusion matrix as a labelled heatmap."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Real')
    ax.set_title('Confusion matrix (test set)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curves(history, path):
    """Save accuracy and loss curves (train vs validation) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(history['loss']) + 1)
    for ax, metric in zip(axes, ('accuracy', 'loss')):
        ax.plot(epochs, history[metric], label='train', linewidth=2)
        ax.plot(epochs, history[f'val_{metric}'], label='validation', linewidth=2)
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_reports(results, reports_dir):
    """Write metrics.json and confusion_matrix.png to ``reports_dir``."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / 'metrics.json').write_text(json.dumps(results, indent=2))
    save_confusion_matrix(np.array(results['confusion_matrix']), reports_dir / 'confusion_matrix.png')


def print_summary(results):
    """Print accuracy, loss and per-class precision/recall/F1."""
    print(f"Test accuracy: {results['test_accuracy'] * 100:.2f}%  "
          f"loss: {results['test_loss']:.4f}  (n={results['n_test']})")
    report = results['classification_report']
    for name in CLASS_NAMES:
        r = report[name]
        print(f"  {name:11s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  f1 {r['f1-score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a BrainScan model on the test set.')
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--image-dir', type=Path, default=RESIZED_DIR,
                        help='Resized images (created by brainscan-train).')
    parser.add_argument('--reports-dir', type=Path, default=REPORTS_DIR)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    images, labels = load_dataset(args.image_dir)
    _, X_test, _, y_test = split_dataset(images, labels, seed=args.seed)
    results = evaluate_model(tf.keras.models.load_model(args.model), X_test, y_test)
    write_reports(results, args.reports_dir)
    print_summary(results)
    print(f'Reports written to {args.reports_dir}')


if __name__ == '__main__':
    main()
