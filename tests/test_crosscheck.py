import numpy as np
import pytest

from sacaavf.analytic import simulate_layer as analytic_simulate
from sacaavf.cyclesim import simulate_layer as cycle_simulate


ARRAYS = [(2, 2), (3, 3), (2, 4), (5, 5)]


def _case(rng: np.random.Generator, n: int, k: int, m: int):
    I = rng.integers(-2, 3, size=(n, k), dtype=np.int32).astype(np.float32)
    W = rng.integers(-2, 3, size=(k, m), dtype=np.int32).astype(np.float32)
    I[rng.random(I.shape) < 0.5] = 0.0
    W[rng.random(W.shape) < 0.5] = 0.0
    I[0, :] = 0.0
    W[:, 0] = 0.0
    return I, W


@pytest.mark.parametrize("arr_h,arr_w", ARRAYS)
def test_cycle_and_analytic_simulators_match(arr_h, arr_w):
    rng = np.random.default_rng(0x5ACA + arr_h * 17 + arr_w)
    shapes = [
        (1, max(1, arr_h - 1), max(1, arr_w - 1)),
        (arr_h + 2, arr_h + 1, arr_w + 2),
    ]
    for n, k, m in shapes:
        I, W = _case(rng, n, k, m)
        expected = I @ W
        for dataflow in ("WS", "IS", "OS"):
            analytic = analytic_simulate(I, W, dataflow, arr_h, arr_w)
            cyclesim = cycle_simulate(I, W, dataflow, arr_h, arr_w)
            np.testing.assert_allclose(cyclesim.O, expected)
            assert cyclesim.cycles == analytic.cycles
            assert cyclesim.ace_bits == analytic.ace_bits
            np.testing.assert_array_equal(cyclesim.ace_per_pe, analytic.ace_per_pe)
            assert cyclesim.ace_by_reg == analytic.ace_by_reg
            assert cyclesim.active_pe_cycles == analytic.active_pe_cycles


def test_ws_pre_store_weight_ace_counts():
    I = np.ones((1, 3), dtype=np.float32)
    W = np.ones((3, 3), dtype=np.float32)
    result = cycle_simulate(I, W, "WS", 3, 3)
    pre_store_counts = [
        int(mask[:, :, 1].sum()) for mask in result.ace_masks[:3]
    ]
    assert pre_store_counts == [3, 6, 9]
