from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["benign", "malignant", "normal"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(data_dir: Path) -> tuple[list[str], list[int]]:
    image_paths: list[str] = []
    labels: list[int] = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(label_index)
    if not image_paths:
        expected = data_dir / "{benign,malignant,normal}"
        raise FileNotFoundError(
            f"No ultrasound images found. Expected class folders such as {expected}."
        )
    return image_paths, labels


def make_datasets(
    image_paths: list[str],
    labels: list[int],
    image_size: tuple[int, int],
    batch_size: int,
    random_state: int,
):
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    def load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32), label

    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(len(train_paths), seed=random_state)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_paths, validation_labels)
    )
    validation_dataset = validation_dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, validation_dataset, train_labels


def build_model(image_size: tuple[int, int], learning_rate: float) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="safe_image_augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*image_size, 3))
    augmented = augmentation(inputs)
    scaled = tf.keras.applications.mobilenet_v2.preprocess_input(augmented)
    features = base(scaled, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.35)(features)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.random_state)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    image_size = (args.image_size, args.image_size)
    image_paths, labels = find_images(data_dir)
    train_dataset, validation_dataset, train_labels = make_datasets(
        image_paths, labels, image_size, args.batch_size, args.random_state
    )
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    total = class_counts.sum()
    class_weights = {
        index: float(total / (len(CLASS_NAMES) * count))
        for index, count in enumerate(class_counts)
        if count
    }

    model = build_model(image_size, args.learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    y_true = np.concatenate([labels.numpy() for _, labels in validation_dataset])
    y_probability = model.predict(validation_dataset, verbose=0)
    y_predicted = y_probability.argmax(axis=1)
    report = classification_report(
        y_true,
        y_predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    model_path = output_dir / "ultrasound_model.keras"
    metrics_path = output_dir / "ultrasound_metrics.json"
    model.save(model_path)
    metrics = {
        "dataset": "MammoBench v2 BUSI ultrasound subset",
        "classes": CLASS_NAMES,
        "image_size": list(image_size),
        "image_count": len(image_paths),
        "validation_count": len(y_true),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true, y_predicted, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Images used: {len(image_paths)}")
    print(f"Validation accuracy: {report['accuracy']:.3f}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print("Educational demonstration only; not for medical diagnosis.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ultrasound image classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ultrasound/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())