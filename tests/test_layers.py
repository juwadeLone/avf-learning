import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from sacaavf.layers import extract_layer_matrices, layer_preactivation
from sacaavf.models import build_cifar10_cnn, build_lenet5, build_vgg16


@pytest.mark.parametrize(
    "builder,shape",
    [
        (build_lenet5, (1, 28, 28, 1)),
        (build_cifar10_cnn, (1, 32, 32, 3)),
        (lambda: build_vgg16(weights=None), (1, 224, 224, 3)),
    ],
)
def test_extracted_matrices_reproduce_layer_preactivations(builder, shape):
    model = builder()
    x = np.random.default_rng(123).normal(size=shape).astype(np.float32)
    targets = [
        layer
        for layer in model.layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))
    ]
    input_model = keras.Model(model.inputs, [layer.input for layer in targets])
    activations = input_model(x, training=False)
    if not isinstance(activations, (list, tuple)):
        activations = [activations]
    matrices = extract_layer_matrices(model, x)
    assert [name for name, _, _ in matrices] == [layer.name for layer in targets]

    for layer, activation, (_, i_matrix, w_matrix) in zip(
        targets, activations, matrices
    ):
        expected = layer_preactivation(layer, activation.numpy())
        reconstructed = i_matrix @ w_matrix + layer.bias.numpy()
        if isinstance(layer, keras.layers.Conv2D):
            reconstructed = reconstructed.reshape(expected.shape)
        np.testing.assert_allclose(reconstructed, expected, atol=1e-3, rtol=1e-3)
