"""Closed-form timing and ACE accounting for tiled systolic-array layers."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


BITS = 32


@dataclass
class LayerResult:
    cycles: int
    ace_reg_cycles: int
    ace_per_pe: np.ndarray
    active_pe_cycles: int
    ace_by_reg: tuple[int, int, int]
    O: np.ndarray | None = None
    ace_masks: list[np.ndarray] = field(default_factory=list)


def _normalize_dataflow(dataflow: str) -> str:
    value = str(dataflow).upper()
    if value not in {"WS", "IS", "OS"}:
        raise ValueError(f"unsupported dataflow: {dataflow!r}")
    return value


def _validate_inputs(
    I: np.ndarray, W: np.ndarray, arr_h: int, arr_w: int
) -> tuple[np.ndarray, np.ndarray]:
    I = np.asarray(I, dtype=np.float32)
    W = np.asarray(W, dtype=np.float32)
    if I.ndim != 2 or W.ndim != 2:
        raise ValueError("I and W must be rank-2 arrays")
    if I.shape[1] != W.shape[0]:
        raise ValueError(f"inner dimensions do not match: {I.shape} and {W.shape}")
    if arr_h <= 0 or arr_w <= 0:
        raise ValueError("array dimensions must be positive")
    return I, W


def simulate_layer(
    I: np.ndarray, W: np.ndarray, dataflow: str, arr_h: int, arr_w: int
) -> LayerResult:
    """Simulate one matrix multiplication with closed-form ACE counts.

    The physical array is reused for every tile, so ``ace_per_pe`` accumulates
    ACE bits at each physical PE across all tiles.
    """

    I, W = _validate_inputs(I, W, arr_h, arr_w)
    dataflow = _normalize_dataflow(dataflow)
    N, K = I.shape
    _, M = W.shape
    if min(N, K, M) <= 0:
        raise ValueError("I and W must have non-zero dimensions")

    ace_per_pe = np.zeros((arr_h, arr_w), dtype=np.int64)
    cycles = 0
    active_pe_cycles = 0
    ace_by_reg = [0, 0, 0]

    if dataflow == "WS":
        nz_i_col = np.count_nonzero(I, axis=0)
        for m0 in range(0, M, arr_w):
            mw = min(arr_w, M - m0)
            for k0 in range(0, K, arr_h):
                kh = min(arr_h, K - k0)
                tile_cycles = 2 * kh + N + mw - 2
                cycles += tile_cycles
                active_pe_cycles += kh * mw * tile_cycles

                i_idx = np.arange(kh)[:, None]
                j_idx = np.arange(mw)[None, :]
                weight_mac_ace = i_idx + j_idx + nz_i_col[k0 : k0 + kh, None]
                weight_ace = (kh - i_idx) + weight_mac_ace
                ifmap_ace = N * (W[k0 : k0 + kh, m0 : m0 + mw] != 0)
                psum_ace = np.full((kh, mw), N, dtype=np.int64)
                ace_per_pe[:kh, :mw] += weight_ace + ifmap_ace + psum_ace

                ace_by_reg[0] += int(ifmap_ace.sum())
                ace_by_reg[1] += int(weight_mac_ace.sum())
                ace_by_reg[2] += int(psum_ace.sum())
                ace_by_reg[1] += mw * kh * (kh + 1) // 2

    elif dataflow == "IS":
        nz_w_row = np.count_nonzero(W, axis=1)
        for n0 in range(0, N, arr_w):
            nw = min(arr_w, N - n0)
            for k0 in range(0, K, arr_h):
                kh = min(arr_h, K - k0)
                tile_cycles = 2 * kh + M + nw - 2
                cycles += tile_cycles
                active_pe_cycles += kh * nw * tile_cycles

                i_idx = np.arange(kh)[:, None]
                j_idx = np.arange(nw)[None, :]
                ifmap_mac_ace = i_idx + j_idx + nz_w_row[k0 : k0 + kh, None]
                ifmap_ace = (kh - i_idx) + ifmap_mac_ace
                weight_ace = M * (
                    I[n0 : n0 + nw, k0 : k0 + kh].T != 0
                )
                psum_ace = np.full((kh, nw), M, dtype=np.int64)
                ace_per_pe[:kh, :nw] += ifmap_ace + weight_ace + psum_ace

                ace_by_reg[0] += int(ifmap_mac_ace.sum())
                ace_by_reg[1] += int(weight_ace.sum())
                ace_by_reg[2] += int(psum_ace.sum())
                ace_by_reg[0] += nw * kh * (kh + 1) // 2

    else:
        nz_w_col = np.count_nonzero(W, axis=0)
        nz_i_row = np.count_nonzero(I, axis=1)
        for n0 in range(0, N, arr_h):
            nh = min(arr_h, N - n0)
            for m0 in range(0, M, arr_w):
                mw = min(arr_w, M - m0)
                tile_cycles = K + nh + mw - 2
                cycles += tile_cycles
                active_pe_cycles += nh * mw * tile_cycles

                i_idx = np.arange(nh)[:, None]
                j_idx = np.arange(mw)[None, :]
                ifmap_ace = np.broadcast_to(
                    nz_w_col[m0 : m0 + mw][None, :], (nh, mw)
                )
                weight_ace = np.broadcast_to(
                    nz_i_row[n0 : n0 + nh, None], (nh, mw)
                )
                psum_ace = tile_cycles - i_idx - j_idx
                ace_per_pe[:nh, :mw] += ifmap_ace + weight_ace + psum_ace

                ace_by_reg[0] += int(ifmap_ace.sum())
                ace_by_reg[1] += int(weight_ace.sum())
                ace_by_reg[2] += int(psum_ace.sum())

    ace_reg_cycles = int(sum(ace_by_reg))
    return LayerResult(
        cycles=int(cycles),
        ace_reg_cycles=ace_reg_cycles,
        ace_per_pe=ace_per_pe,
        active_pe_cycles=int(active_pe_cycles),
        ace_by_reg=tuple(int(x) for x in ace_by_reg),
        O=I @ W,
    )
