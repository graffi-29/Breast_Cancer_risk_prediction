from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


CLASS_NAMES = ["benign", "malignant"]
SOURCE_NAMES = ["DDSM Dataset", "INbreast Dataset", "MIAS Dataset"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def patient_group(image_path: Path) -> str:
    name = re.sub(r" \(\d+\)$", "", image_path.stem)
    name = name.split(".", maxsplit=1)[0]
    return name


def collect_images(dataset_root: Path) -> tuple[list[str], list[int], list[str]]:
    image_paths: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    for source_name in SOURCE_NAMES:
        for label_index, class_name in enumerate(["Benign Masses", "Malignant Masses"]):
            class_dir = dataset_root / source_name / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing dataset folder: {class_dir}")
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(str(image_path))
                    labels.append(label_index)
                    groups.append(f"{source_name}:{patient_group(image_path)}")
    if not image_paths:
        raise FileNotFoundError("No mammogram images were found.")
    return image_paths, labels, groups


def split_paths(
    image_paths: list[str], labels: list[int], groups: list[str], random_state: int
) -> tuple[list[str], list[str], list[int], list[int]]:
    indices = np.arange(len(image_paths))
    first_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_indices, test_indices = next(first_split.split(indices, labels, groups))
    train_groups = [groups[index] for index in train_indices]
    second_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state + 1)
    inner_train, validation_indices = next(
        second_split.split(train_indices, np.asarray(labels)[train_indices], train_groups)
    )
    final_train_indices = train_indices[inner_train]
    return (
        [image_paths[index] for index in final_train_indices],
        [image_paths[index] for index in train_indices[validation_indices]],
        [labels[index] for index in final_train_indices],
        [labels[index] for index in train_indices[validation_indices]],
    )


def make_dataset(
    image_paths: list[str],
    labels: list[int],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    def load_image(path: tf.Tensor, label: tf.Tensor):
        image = tf.image.decode_png(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, image_size)
        return tf.cast(image, tf.float32), label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(image_paths), seed=42, reshuffle_each_iteration=True)
    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(image_size: tuple[int, int], learning_rate: float) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet", name="image_backbone"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*image_size, 3))
    augmented = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
        ]
    )(inputs)
    scaled = tf.keras.applications.mobilenet_v2.preprocess_input(augmented)
    features = base(scaled, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.35)(features)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.random_state)
    image_size = (args.image_size, args.image_size)
    image_paths, labels, groups = collect_images(Path(args.dataset_root))
    train_paths, validation_paths, train_labels, validation_labels = split_paths(
        image_paths, labels, groups, args.random_state
    )
    train_dataset = make_dataset(train_paths, train_labels, image_size, args.batch_size, shuffle=True)
    validation_dataset = make_dataset(
        validation_paths, validation_labels, image_size, args.batch_size
    )
    counts = np.bincount(train_labels, minlength=2)
    total = counts.sum()
    class_weights = {index: float(total / (2 * count)) for index, count in enumerate(counts) if count}
    model = build_model(image_size, args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
    )
    if args.fine_tune_epochs > 0:
        backbone = model.get_layer("image_backbone")
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
        )
        for key, values in fine_tune_history.history.items():
            history.history.setdefault(key, []).extend(values)

    true_labels = np.asarray(validation_labels)
    probabilities = model.predict(validation_dataset, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    report = classification_report(
        true_labels, predictions, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    metrics = {
        "dataset": "DDSM, INbreast, and MIAS mammogram image folders",
        "classes": CLASS_NAMES,
        "image_count": len(image_paths),
        "train_count": len(train_paths),
        "validation_count": len(validation_paths),
        "positive_class": "malignant",
        "validation_roc_auc": float(roc_auc_score(true_labels, probabilities)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "image_size": list(image_size),
        "random_state": args.random_state,
    }
    model.save(output_dir / "mammogram_model.keras")
    (output_dir / "mammogram_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Images used: {len(image_paths)}")
    print(f"Grouped validation ROC-AUC: {metrics['validation_roc_auc']:.3f}")
    print(f"Malignant recall: {report['malignant']['recall']:.3f}")
    print(f"Saved model: {output_dir / 'mammogram_model.keras'}")
    print("Educational demonstration only; not for medical diagnosis.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a grouped mammogram classifier.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_mammogram"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())