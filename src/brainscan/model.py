"""CNN architecture used to classify brain MRI scans."""

import tensorflow as tf

from brainscan.config import CLASS_NAMES, TARGET_SIZE


def build_model(num_classes=len(CLASS_NAMES)):
    """Build and compile the BrainScan CNN.

    Five Conv2D -> BatchNorm -> MaxPooling blocks (32 to 512 filters), then a
    Dense(128) + Dropout(0.5) head with a softmax output. Data augmentation and
    the 0-255 -> 0-1 rescaling are layers of the model, so raw uint8 BGR
    images can be fed directly, both in training and in prediction.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name='data_augmentation')

    conv_blocks = []
    for filters in (32, 64, 128, 256, 512):
        conv_blocks += [
            tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
        ]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(*TARGET_SIZE, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1. / 255),
        *conv_blocks,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
