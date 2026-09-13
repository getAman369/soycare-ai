"""Train an EfficientNetB0 soybean leaf classifier from data/processed."""
from pathlib import Path
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "soybean_disease_model.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def make_datasets():
    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True)
    validation = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    test = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False)
    return train.prefetch(tf.data.AUTOTUNE), validation.prefetch(tf.data.AUTOTUNE), test.prefetch(tf.data.AUTOTUNE), train.class_names


def build_model(class_count):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1), tf.keras.layers.RandomContrast(0.1),
    ])
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    model = tf.keras.Sequential([augmentation, base, tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3), tf.keras.layers.Dense(class_count, activation="softmax")])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])
    return model, base


if __name__ == "__main__":
    train, validation, test, class_names = make_datasets()
    model, base = build_model(len(class_names))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)]
    model.fit(train, validation_data=validation, epochs=15, callbacks=callbacks)
    base.trainable = True
    for layer in base.layers[:-20]: layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=validation, epochs=8, callbacks=callbacks)
    print("Test metrics:", dict(zip(model.metrics_names, model.evaluate(test))))
    print("Class order:", class_names)
