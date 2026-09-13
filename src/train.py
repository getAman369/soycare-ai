"""
Soybean Leaf Disease Model Training Script
Trains an EfficientNetB0 transfer learning architecture with data augmentation,
fine-tuning, and automatic checkpointing.
"""
from pathlib import Path
import tensorflow as tf
import logging

from config import (
    DATA_DIR,
    MODEL_PATH,
    CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    OUTPUTS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("soycare.train")


def create_datasets():
    """Loads train, validation, and test datasets with caching and prefetching."""
    processed_dir = DATA_DIR / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    logger.info("Loading datasets from %s", processed_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.cache().prefetch(autotune),
        val_ds.cache().prefetch(autotune),
        test_ds.cache().prefetch(autotune),
        train_ds.class_names
    )


def build_transfer_model(num_classes):
    """
    Constructs an EfficientNetB0 transfer learning model with an augmentation pipeline,
    global pooling, dropout regularization, and dense softmax classification.
    """
    # Data Augmentation layer to improve generalization on field photography
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.1)
    ], name="data_augmentation")

    # Pretrained EfficientNetB0 backbone (ImageNet weights)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False  # Freeze backbone for initial head training

    # Top classification head
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="SoyCare_EfficientNetB0")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model, base_model


def train_pipeline(epochs_stage1=15, epochs_stage2=10):
    """Executes the two-stage transfer learning and fine-tuning training workflow."""
    train_ds, val_ds, test_ds, class_names = create_datasets()
    logger.info("Found %d classes: %s", len(class_names), class_names)

    model, base_model = build_transfer_model(len(class_names))
    model.summary()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
    ]

    # Stage 1: Feature Extraction (Train top classification layers)
    logger.info("Starting Stage 1: Feature Extraction (%d epochs)", epochs_stage1)
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks
    )

    # Stage 2: Fine-Tuning (Unfreeze top convolutional layers of backbone)
    logger.info("Starting Stage 2: Fine-Tuning backbone top layers")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1 + epochs_stage2,
        initial_epoch=len(history_stage1.history["loss"]),
        callbacks=callbacks
    )

    # Evaluate on held-out test split
    logger.info("Evaluating final model on test split...")
    test_metrics = model.evaluate(test_ds, return_dict=True)
    logger.info("Final Test Performance: %s", test_metrics)

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 Soybean Disease Classifier")
    parser.add_argument("--epochs1", type=int, default=10, help="Epochs for Stage 1 (Feature extraction)")
    parser.add_argument("--epochs2", type=int, default=5, help="Epochs for Stage 2 (Fine-tuning)")
    args = parser.parse_args()

    train_pipeline(epochs_stage1=args.epochs1, epochs_stage2=args.epochs2)
