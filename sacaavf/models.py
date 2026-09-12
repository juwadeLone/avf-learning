"""Keras model and dataset helpers for the Saca-AVF experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def configure_cpu_threads(num_threads: int = 8) -> None:
    """Configure TensorFlow before substantial graph execution."""

    try:
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    except RuntimeError:
        pass


def build_lenet5() -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = keras.layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs)
    x = keras.layers.Conv2D(64, 3, activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(name="pool1")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", name="conv3")(x)
    x = keras.layers.MaxPooling2D(name="pool2")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense1")(x)
    x = keras.layers.Dense(84, activation="relu", name="dense2")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="classifier")(x)
    return keras.Model(inputs, outputs, name="LeNet-5")


def build_cifar10_cnn() -> keras.Model:
    inputs = keras.Input(shape=(32, 32, 3), name="image")
    x = keras.layers.Conv2D(32, 3, activation="relu", name="conv1")(inputs)
    x = keras.layers.Conv2D(32, 3, activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(name="pool1")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", name="conv3")(x)
    x = keras.layers.Conv2D(64, 3, activation="relu", name="conv4")(x)
    x = keras.layers.MaxPooling2D(name="pool2")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(512, activation="relu", name="dense1")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="classifier")(x)
    return keras.Model(inputs, outputs, name="Cifar-10 CNN")


def build_vgg16(weights: str | None = "imagenet") -> keras.Model:
    return keras.applications.VGG16(weights=weights, include_top=True)


def _compile(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if name == "LeNet-5":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train[..., None].astype(np.float32) / 255.0
        x_test = x_test[..., None].astype(np.float32) / 255.0
    elif name == "Cifar-10 CNN":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]
    else:
        raise ValueError(f"no classification dataset for {name}")
    return x_train, y_train.astype(np.int64), x_test, y_test.astype(np.int64)


def load_or_train(
    name: str, *, epochs: int | None = None, verbose: int = 2
) -> tuple[keras.Model, float]:
    """Load a cached model or train it, returning test accuracy in percent."""

    configure_cpu_threads()
    filename = {
        "LeNet-5": "lenet5.keras",
        "Cifar-10 CNN": "cifar10_cnn.keras",
        "VGG-16": "vgg16_imagenet.keras",
    }[name]
    path = MODEL_DIR / filename
    if name == "VGG-16":
        if path.exists():
            model = keras.models.load_model(path)
        else:
            model = build_vgg16()
            model.save(path)
        return model, float("nan")

    x_train, y_train, x_test, y_test = _dataset(name)
    if path.exists():
        model = keras.models.load_model(path)
    else:
        model = _compile(build_lenet5() if name == "LeNet-5" else build_cifar10_cnn())
        if epochs is None:
            epochs = 2 if name == "LeNet-5" else 10
        model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=verbose)
        model.save(path)
    if model.optimizer is None:
        _compile(model)
    _, accuracy = model.evaluate(x_test, y_test, batch_size=256, verbose=0)
    return model, float(accuracy * 100.0)


def load_experiment_images(
    name: str, count: int = 100, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return fixed-seed experiment images, labels when they are compatible."""

    rng = np.random.default_rng(seed)
    if name in {"LeNet-5", "Cifar-10 CNN"}:
        _, _, x_test, y_test = _dataset(name)
        count = min(count, len(x_test))
        indices = rng.choice(len(x_test), size=count, replace=False)
        return x_test[indices], y_test[indices], "MNIST test" if name == "LeNet-5" else "CIFAR-10 test"

    try:
        import tensorflow_datasets as tfds

        dataset, info = tfds.load(
            "imagenette/320px-v2",
            split="train",
            as_supervised=True,
            with_info=True,
            shuffle_files=False,
            download=True,
        )
        images = []
        labels = []
        for image, label in tfds.as_numpy(dataset.take(count)):
            image = tf.image.resize(image, (224, 224)).numpy()
            images.append(keras.applications.vgg16.preprocess_input(image))
            labels.append(int(label))
        return np.asarray(images, dtype=np.float32), np.asarray(labels), "Imagenette train"
    except Exception as exc:
        print(f"VGG natural-image download failed ({exc}); using CIFAR-10 fallback.")
        _, _, x_test, y_test = _dataset("Cifar-10 CNN")
        count = min(count, len(x_test))
        indices = rng.choice(len(x_test), size=count, replace=False)
        images = tf.image.resize(x_test[indices] * 255.0, (224, 224)).numpy()
        return (
            keras.applications.vgg16.preprocess_input(images).astype(np.float32),
            y_test[indices],
            "CIFAR-10 test upscaled fallback",
        )
