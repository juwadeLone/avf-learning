"""Literal cycle-by-cycle reference simulator for the three dataflows."""

from typing import Any

import numpy as np

from .analytic import LayerResult, _normalize_dataflow, _validate_inputs


def _new_mask(arr_h: int, arr_w: int) -> np.ndarray:
    return np.zeros((arr_h, arr_w, 3), dtype=bool)


def _shift_right(
    values: np.ndarray, valid: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray]:
    next_values = np.zeros_like(values)
    next_valid = np.zeros_like(valid)
    if width > 1:
        next_values[:, 1:width] = values[:, : width - 1]
        next_valid[:, 1:width] = valid[:, : width - 1]
    return next_values, next_valid


def _shift_down(
    values: np.ndarray, valid: np.ndarray, height: int
) -> tuple[np.ndarray, np.ndarray]:
    next_values = np.zeros_like(values)
    next_valid = np.zeros_like(valid)
    if height > 1:
        next_values[1:height, :] = values[: height - 1, :]
        next_valid[1:height, :] = valid[: height - 1, :]
    return next_values, next_valid


def _simulate_ws_tile(
    I: np.ndarray,
    W: np.ndarray,
    n0: int,
    k0: int,
    m0: int,
    kh: int,
    mw: int,
    arr_h: int,
    arr_w: int,
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, int]]:
    N = I.shape[0]
    tile_cycles = 2 * kh + N + mw - 2
    ifmap = np.zeros((arr_h, arr_w), dtype=np.float32)
    ifmap_valid = np.zeros((arr_h, arr_w), dtype=bool)
    weight = np.zeros((arr_h, arr_w), dtype=np.float32)
    psum_pipe = np.zeros((arr_h, arr_w), dtype=np.float32)
    psum_valid = np.zeros((arr_h, arr_w), dtype=bool)
    output = np.zeros((N, mw), dtype=np.float32)
    masks: list[np.ndarray] = []
    by_reg = [0, 0, 0]

    for c in range(kh):
        weight[c, :mw] = W[k0 + c, m0 : m0 + mw]
        mask = _new_mask(arr_h, arr_w)
        mask[: c + 1, :mw, 1] = True
        masks.append(mask)
        by_reg[1] += int(mask[:, :, 1].sum())

    for p in range(tile_cycles - kh):
        for i in range(kh):
            n = p - i
            if 0 <= n < N:
                ifmap[i, 0] = I[n, k0 + i]
                ifmap_valid[i, 0] = True

        mask = _new_mask(arr_h, arr_w)
        next_psum = np.zeros_like(psum_pipe)
        next_psum_valid = np.zeros_like(psum_valid)
        for i in range(kh):
            for j in range(mw):
                if p < i + j:
                    mask[i, j, 1] = True
                if ifmap_valid[i, j]:
                    ivalue = ifmap[i, j]
                    wvalue = weight[i, j]
                    partial = psum_pipe[i, j] if psum_valid[i, j] else 0.0
                    result = partial + ivalue * wvalue
                    mask[i, j, 0] = wvalue != 0
                    mask[i, j, 1] |= ivalue != 0
                    mask[i, j, 2] = True
                    if i + 1 < kh:
                        next_psum[i + 1, j] = result
                        next_psum_valid[i + 1, j] = True
                    if i == kh - 1:
                        output[p - i - j, j] = result

        by_reg[0] += int(mask[:, :, 0].sum())
        by_reg[1] += int(mask[:, :, 1].sum())
        by_reg[2] += int(mask[:, :, 2].sum())
        masks.append(mask)
        psum_pipe = next_psum
        psum_valid = next_psum_valid
        ifmap, ifmap_valid = _shift_right(ifmap, ifmap_valid, mw)

    return output, masks, tuple(by_reg)


def _simulate_is_tile(
    I: np.ndarray,
    W: np.ndarray,
    n0: int,
    k0: int,
    mw: int,
    kh: int,
    arr_h: int,
    arr_w: int,
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, int]]:
    M = W.shape[1]
    tile_cycles = 2 * kh + M + mw - 2
    ifmap = np.zeros((arr_h, arr_w), dtype=np.float32)
    weight = np.zeros((arr_h, arr_w), dtype=np.float32)
    weight_valid = np.zeros((arr_h, arr_w), dtype=bool)
    psum_pipe = np.zeros((arr_h, arr_w), dtype=np.float32)
    psum_valid = np.zeros((arr_h, arr_w), dtype=bool)
    output = np.zeros((mw, M), dtype=np.float32)
    masks: list[np.ndarray] = []
    by_reg = [0, 0, 0]

    for c in range(kh):
        ifmap[c, :mw] = I[n0 : n0 + mw, k0 + c]
        mask = _new_mask(arr_h, arr_w)
        mask[: c + 1, :mw, 0] = True
        masks.append(mask)
        by_reg[0] += int(mask[:, :, 0].sum())

    for p in range(tile_cycles - kh):
        for i in range(kh):
            m = p - i
            if 0 <= m < M:
                weight[i, 0] = W[k0 + i, m]
                weight_valid[i, 0] = True

        mask = _new_mask(arr_h, arr_w)
        next_psum = np.zeros_like(psum_pipe)
        next_psum_valid = np.zeros_like(psum_valid)
        for i in range(kh):
            for j in range(mw):
                if p < i + j:
                    mask[i, j, 0] = True
                if weight_valid[i, j]:
                    ivalue = ifmap[i, j]
                    wvalue = weight[i, j]
                    partial = psum_pipe[i, j] if psum_valid[i, j] else 0.0
                    result = partial + ivalue * wvalue
                    mask[i, j, 0] |= wvalue != 0
                    mask[i, j, 1] = ivalue != 0
                    mask[i, j, 2] = True
                    if i + 1 < kh:
                        next_psum[i + 1, j] = result
                        next_psum_valid[i + 1, j] = True
                    if i == kh - 1:
                        output[j, p - i - j] = result

        by_reg[0] += int(mask[:, :, 0].sum())
        by_reg[1] += int(mask[:, :, 1].sum())
        by_reg[2] += int(mask[:, :, 2].sum())
        masks.append(mask)
        psum_pipe = next_psum
        psum_valid = next_psum_valid
        weight, weight_valid = _shift_right(weight, weight_valid, mw)

    return output, masks, tuple(by_reg)


def _simulate_os_tile(
    I: np.ndarray,
    W: np.ndarray,
    n0: int,
    m0: int,
    nh: int,
    mw: int,
    arr_h: int,
    arr_w: int,
) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, int]]:
    K = I.shape[1]
    tile_cycles = K + nh + mw - 2
    ifmap = np.zeros((arr_h, arr_w), dtype=np.float32)
    ifmap_valid = np.zeros((arr_h, arr_w), dtype=bool)
    weight = np.zeros((arr_h, arr_w), dtype=np.float32)
    weight_valid = np.zeros((arr_h, arr_w), dtype=bool)
    psum = np.zeros((arr_h, arr_w), dtype=np.float32)
    output = np.zeros((nh, mw), dtype=np.float32)
    masks: list[np.ndarray] = []
    by_reg = [0, 0, 0]

    for p in range(tile_cycles):
        for i in range(nh):
            k = p - i
            if 0 <= k < K:
                ifmap[i, 0] = I[n0 + i, k]
                ifmap_valid[i, 0] = True
        for j in range(mw):
            k = p - j
            if 0 <= k < K:
                weight[0, j] = W[k, m0 + j]
                weight_valid[0, j] = True

        mask = _new_mask(arr_h, arr_w)
        for i in range(nh):
            for j in range(mw):
                if ifmap_valid[i, j] and weight_valid[i, j]:
                    ivalue = ifmap[i, j]
                    wvalue = weight[i, j]
                    psum[i, j] += ivalue * wvalue
                    mask[i, j, 0] = wvalue != 0
                    mask[i, j, 1] = ivalue != 0
        for i in range(nh):
            for j in range(mw):
                if p >= i + j:
                    mask[i, j, 2] = True
        by_reg[0] += int(mask[:, :, 0].sum())
        by_reg[1] += int(mask[:, :, 1].sum())
        by_reg[2] += int(mask[:, :, 2].sum())
        masks.append(mask)

        ifmap, ifmap_valid = _shift_right(ifmap, ifmap_valid, mw)
        weight, weight_valid = _shift_down(weight, weight_valid, nh)

    output[:, :] = psum[:nh, :mw]
    return output, masks, tuple(by_reg)


def simulate_layer(
    I: np.ndarray, W: np.ndarray, dataflow: str, arr_h: int, arr_w: int
) -> LayerResult:
    """Run a literal one-hop-per-cycle reference simulation."""

    I, W = _validate_inputs(I, W, arr_h, arr_w)
    dataflow = _normalize_dataflow(dataflow)
    N, K = I.shape
    _, M = W.shape
    O = np.zeros((N, M), dtype=np.float32)
    ace_per_pe = np.zeros((arr_h, arr_w), dtype=np.int64)
    cycles = 0
    active_pe_cycles = 0
    ace_by_reg = [0, 0, 0]
    all_masks: list[np.ndarray] = []

    if dataflow == "WS":
        for m0 in range(0, M, arr_w):
            mw = min(arr_w, M - m0)
            for k0 in range(0, K, arr_h):
                kh = min(arr_h, K - k0)
                tile_o, masks, tile_regs = _simulate_ws_tile(
                    I, W, 0, k0, m0, kh, mw, arr_h, arr_w
                )
                O[:, m0 : m0 + mw] += tile_o
                tile_cycles = 2 * kh + N + mw - 2
                cycles += tile_cycles
                active_pe_cycles += kh * mw * tile_cycles
                all_masks.extend(masks)
                ace_by_reg = [a + b for a, b in zip(ace_by_reg, tile_regs)]
                for mask in masks:
                    ace_per_pe += mask.sum(axis=2, dtype=np.int64)

    elif dataflow == "IS":
        for n0 in range(0, N, arr_w):
            nw = min(arr_w, N - n0)
            for k0 in range(0, K, arr_h):
                kh = min(arr_h, K - k0)
                tile_o, masks, tile_regs = _simulate_is_tile(
                    I, W, n0, k0, nw, kh, arr_h, arr_w
                )
                O[n0 : n0 + nw, :] += tile_o
                tile_cycles = 2 * kh + M + nw - 2
                cycles += tile_cycles
                active_pe_cycles += kh * nw * tile_cycles
                all_masks.extend(masks)
                ace_by_reg = [a + b for a, b in zip(ace_by_reg, tile_regs)]
                for mask in masks:
                    ace_per_pe += mask.sum(axis=2, dtype=np.int64)

    else:
        for n0 in range(0, N, arr_h):
            nh = min(arr_h, N - n0)
            for m0 in range(0, M, arr_w):
                mw = min(arr_w, M - m0)
                tile_o, masks, tile_regs = _simulate_os_tile(
                    I, W, n0, m0, nh, mw, arr_h, arr_w
                )
                O[n0 : n0 + nh, m0 : m0 + mw] = tile_o
                tile_cycles = K + nh + mw - 2
                cycles += tile_cycles
                active_pe_cycles += nh * mw * tile_cycles
                all_masks.extend(masks)
                ace_by_reg = [a + b for a, b in zip(ace_by_reg, tile_regs)]
                for mask in masks:
                    ace_per_pe += mask.sum(axis=2, dtype=np.int64)

    return LayerResult(
        cycles=int(cycles),
        ace_bits=int(sum(ace_by_reg)),
        ace_per_pe=ace_per_pe,
        active_pe_cycles=int(active_pe_cycles),
        ace_by_reg=tuple(int(x) for x in ace_by_reg),
        O=O,
        ace_masks=all_masks,
    )
