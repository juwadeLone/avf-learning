"""Extract im2col matrices from actual Keras layer activations."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras


def _activation_model(model: keras.Model) -> keras.Model:
    layers = [
        layer
        for layer in model.layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))
    ]
    return keras.Model(model.inputs, [layer.input for layer in layers])


def extract_layer_matrices(
    model: keras.Model, x: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return ``(layer name, I, W)`` for every Conv2D and Dense layer."""

    targets = [
        layer
        for layer in model.layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))
    ]
    if not targets:
        return []
    inputs = np.asarray(x, dtype=np.float32)
    if inputs.ndim == len(model.input_shape) - 1:
        inputs = inputs[None, ...]
    activation_model = _activation_model(model)
    activations = activation_model(inputs, training=False)
    if not isinstance(activations, (list, tuple)):
        activations = [activations]

    output: list[tuple[str, np.ndarray, np.ndarray]] = []
    for layer, activation in zip(targets, activations):
        activation = tf.convert_to_tensor(activation)
        kernel = np.asarray(layer.kernel.numpy(), dtype=np.float32)
        if isinstance(layer, keras.layers.Conv2D):
            strides = (1, layer.strides[0], layer.strides[1], 1)
            rates = (1, layer.dilation_rate[0], layer.dilation_rate[1], 1)
            patches = tf.image.extract_patches(
                activation,
                sizes=(1, kernel.shape[0], kernel.shape[1], 1),
                strides=strides,
                rates=rates,
                padding=layer.padding.upper(),
            )
            i_matrix = tf.reshape(patches[0], (-1, kernel.shape[0] * kernel.shape[1] * kernel.shape[2]))
            w_matrix = kernel.reshape((-1, kernel.shape[-1]))
        else:
            i_matrix = tf.reshape(activation, (1, -1))
            w_matrix = kernel.reshape((kernel.shape[0], kernel.shape[1]))
        output.append(
            (
                layer.name,
                np.asarray(i_matrix.numpy(), dtype=np.float32),
                np.asarray(w_matrix, dtype=np.float32),
            )
        )
    return output


def layer_preactivation(layer: keras.layers.Layer, activation: np.ndarray) -> np.ndarray:
    """Compute a layer's affine output before its configured activation."""

    x = tf.convert_to_tensor(activation, dtype=tf.float32)
    if isinstance(layer, keras.layers.Conv2D):
        return tf.nn.convolution(
            x,
            layer.kernel,
            strides=layer.strides,
            padding=layer.padding.upper(),
            dilations=layer.dilation_rate,
            data_format="NHWC",
        ).numpy() + layer.bias.numpy().reshape((1, 1, 1, -1))
    if isinstance(layer, keras.layers.Dense):
        return (tf.linalg.matmul(tf.reshape(x, (1, -1)), layer.kernel) + layer.bias).numpy()
    raise TypeError(type(layer).__name__)
