"""Run the model/dataflow/array sweep and write SPEC §7 artifacts."""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analytic import simulate_layer
from .layers import extract_layer_matrices
from .models import load_experiment_images, load_or_train


ARRAYS = (32, 64, 128, 256, 512)
DATAFLOWS = ("WS", "IS", "OS")
MODEL_ORDER = ("LeNet-5", "Cifar-10 CNN", "VGG-16")
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _config_key(dataflow: str, size: int) -> str:
    return f"{dataflow}_{size}x{size}"


def _compute_image_summary(
    payload: tuple[str, list[tuple[str, np.ndarray, np.ndarray]], tuple[int, ...]]
) -> dict[str, Any]:
    model_name, matrices, array_sizes = payload
    summary: dict[str, Any] = {"configs": {}, "selected_maps": {}}
    selected = {
        ("LeNet-5", "conv1", "WS", 32): "LeNet-5_L1_WS_32x32",
        ("Cifar-10 CNN", "conv4", "IS", 32): "Cifar-10_CNN_L4_IS_32x32",
        ("VGG-16", "block1_conv1", "OS", 64): "VGG-16_L1_OS_64x64",
    }
    for dataflow in DATAFLOWS:
        for size in array_sizes:
            total_cycles = 0
            total_ace = 0
            total_active = 0
            total_regs = np.zeros(3, dtype=np.int64)
            layers = []
            for layer_name, I, W in matrices:
                result = simulate_layer(I, W, dataflow, size, size)
                total_cycles += result.cycles
                total_ace += result.ace_reg_cycles
                total_active += result.active_pe_cycles
                total_regs += np.asarray(result.ace_by_reg, dtype=np.int64)
                layers.append(
                    {
                        "name": layer_name,
                        "avf": result.ace_reg_cycles / (3.0 * size * size * result.cycles),
                        "cycles": result.cycles,
                    }
                )
                map_key = selected.get((model_name, layer_name, dataflow, size))
                if map_key is not None:
                    summary["selected_maps"][map_key] = (
                        result.ace_per_pe / (3.0 * result.cycles)
                    )
            denom = 3.0 * size * size * total_cycles
            summary["configs"][_config_key(dataflow, size)] = {
                "avf": total_ace / denom,
                "utilization": total_active / (size * size * total_cycles),
                "ace_by_reg": total_regs.tolist(),
                "layers": layers,
            }
    return summary


def _extract_all(model: Any, images: np.ndarray) -> list[list[tuple[str, np.ndarray, np.ndarray]]]:
    return [extract_layer_matrices(model, image) for image in images]


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_model(
    model_name: str,
    summaries: list[dict[str, Any]],
    accuracies: dict[str, float],
    image_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    model_rows = []
    util_rows = []
    layer_rows = []
    reg_rows = []
    maps = {}
    cycle_rows = []
    for dataflow in DATAFLOWS:
        for size in ARRAYS:
            key = _config_key(dataflow, size)
            configs = [summary["configs"][key] for summary in summaries]
            avf_mean, avf_std = _mean_std([x["avf"] for x in configs])
            util_mean, util_std = _mean_std([x["utilization"] for x in configs])
            array_name = f"{size}x{size}"
            model_rows.append(
                {
                    "model": model_name,
                    "dataflow": dataflow,
                    "array": array_name,
                    "mean_avf": avf_mean,
                    "std_avf": avf_std,
                    "accuracy_percent": accuracies.get(model_name, float("nan")),
                    "image_source": image_source,
                    "image_count": len(summaries),
                }
            )
            util_rows.append(
                {
                    "model": model_name,
                    "dataflow": dataflow,
                    "array": array_name,
                    "mean_utilization": util_mean,
                    "std_utilization": util_std,
                    "image_count": len(summaries),
                }
            )
            regs = np.asarray([x["ace_by_reg"] for x in configs], dtype=np.float64).sum(axis=0)
            shares = regs / regs.sum() if regs.sum() else np.zeros(3)
            reg_rows.append(
                {
                    "model": model_name,
                    "dataflow": dataflow,
                    "array": array_name,
                    "ifmap_share": shares[0],
                    "weight_share": shares[1],
                    "psum_share": shares[2],
                }
            )
            layer_names = [item["name"] for item in configs[0]["layers"]]
            for layer_index, layer_name in enumerate(layer_names):
                values = [
                    item["layers"][layer_index]["avf"] for item in configs
                ]
                layer_rows.append(
                    {
                        "model": model_name,
                        "dataflow": dataflow,
                        "array": array_name,
                        "layer": layer_name,
                        "avf": float(np.mean(values)),
                        "std_avf": float(np.std(values)),
                    }
                )
                if model_name == "Cifar-10 CNN" and size == 32:
                    cycle_rows.append(
                        {
                            "model": model_name,
                            "dataflow": dataflow,
                            "array": array_name,
                            "layer": layer_name,
                            "cycles": configs[0]["layers"][layer_index]["cycles"],
                        }
                    )
    for summary in summaries:
        maps.update(summary["selected_maps"])
    return model_rows, util_rows, layer_rows, reg_rows, maps, cycle_rows


def _plot_bars(rows: list[dict[str, Any]], value_key: str, error_key: str, path: Path, ylabel: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    colors = {"WS": "#4c72b0", "IS": "#55a868", "OS": "#c44e52"}
    for axis, model in zip(axes, MODEL_ORDER):
        model_rows = [row for row in rows if row["model"] == model]
        x = np.arange(len(ARRAYS))
        width = 0.24
        for offset, dataflow in enumerate(DATAFLOWS):
            subset = [row for row in model_rows if row["dataflow"] == dataflow]
            values = [100 * row[value_key] for row in subset]
            errors = [100 * row[error_key] for row in subset]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=dataflow,
                color=colors[dataflow],
            )
        axis.set_title(model)
        axis.set_xticks(x, [str(size) for size in ARRAYS])
        axis.set_xlabel("Array size")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_layers(rows: list[dict[str, Any]], path: Path) -> None:
    selections = [("LeNet-5", "64x64"), ("Cifar-10 CNN", "64x64"), ("VGG-16", "32x32")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    colors = {"WS": "#4c72b0", "IS": "#55a868", "OS": "#c44e52"}
    for axis, (model, array_name) in zip(axes, selections):
        subset = [row for row in rows if row["model"] == model and row["array"] == array_name]
        names = []
        for dataflow in DATAFLOWS:
            points = [row for row in subset if row["dataflow"] == dataflow]
            if not names:
                names = [row["layer"] for row in points]
            axis.plot(
                range(len(points)),
                [100 * row["avf"] for row in points],
                marker="o",
                label=dataflow,
                color=colors[dataflow],
            )
        axis.set_title(f"{model} ({array_name})")
        axis.set_xticks(range(len(names)), names, rotation=60, ha="right")
        axis.set_ylabel("Layer AVF (%)")
        axis.grid(alpha=0.25)
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_maps(maps: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, (key, values) in zip(axes, sorted(maps.items())):
        image = axis.imshow(values, cmap="magma", vmin=0)
        axis.set_title(key.replace("_", "\n"))
        axis.set_xlabel("PE column")
        axis.set_ylabel("PE row")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_sweep(
    model_names: tuple[str, ...] = MODEL_ORDER,
    *,
    image_count: int = 100,
    vgg_image_count: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the requested sweep and return accuracy/source/runtime metadata."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_model_rows = []
    all_util_rows = []
    all_layer_rows = []
    all_reg_rows = []
    all_cycle_rows = []
    all_maps: dict[str, np.ndarray] = {}
    metadata = {"accuracies": {}, "image_sources": {}, "image_counts": {}}

    for model_name in model_names:
        model_started = time.perf_counter()
        epochs = 2 if model_name == "LeNet-5" else 10
        model, accuracy = load_or_train(model_name, epochs=epochs)
        count = vgg_image_count if model_name == "VGG-16" else image_count
        images, _, source = load_experiment_images(model_name, count=count, seed=seed)
        metadata["accuracies"][model_name] = accuracy
        metadata["image_sources"][model_name] = source
        metadata["image_counts"][model_name] = len(images)
        matrix_sets = _extract_all(model, images)
        payloads = [(model_name, matrices, ARRAYS) for matrices in matrix_sets]
        if model_name == "VGG-16":
            summaries = [_compute_image_summary(payload) for payload in payloads]
        else:
            processes = min(8, len(payloads))
            context = mp.get_context("fork")
            with context.Pool(processes=processes) as pool:
                summaries = pool.map(_compute_image_summary, payloads)
        model_rows, util_rows, layer_rows, reg_rows, maps, cycle_rows = _aggregate_model(
            model_name, summaries, metadata["accuracies"], source
        )
        all_model_rows.extend(model_rows)
        all_util_rows.extend(util_rows)
        all_layer_rows.extend(layer_rows)
        all_reg_rows.extend(reg_rows)
        all_cycle_rows.extend(cycle_rows)
        all_maps.update(maps)
        print(
            f"{model_name}: accuracy={accuracy:.2f}%, source={source}, "
            f"images={len(images)}, elapsed={time.perf_counter() - model_started:.1f}s"
        )

    _write_csv(
        RESULTS_DIR / "fig6_model_avf.csv",
        ["model", "dataflow", "array", "mean_avf", "std_avf", "accuracy_percent", "image_source", "image_count"],
        all_model_rows,
    )
    _write_csv(
        RESULTS_DIR / "fig7_pe_util.csv",
        ["model", "dataflow", "array", "mean_utilization", "std_utilization", "image_count"],
        all_util_rows,
    )
    _write_csv(
        RESULTS_DIR / "fig12_layer_avf.csv",
        ["model", "dataflow", "array", "layer", "avf", "std_avf"],
        all_layer_rows,
    )
    _write_csv(
        RESULTS_DIR / "ace_by_reg.csv",
        ["model", "dataflow", "array", "ifmap_share", "weight_share", "psum_share"],
        all_reg_rows,
    )
    _write_csv(
        RESULTS_DIR / "cifar_layer_cycles.csv",
        ["model", "dataflow", "array", "layer", "cycles"],
        all_cycle_rows,
    )
    np.savez(RESULTS_DIR / "fig11_pe_maps.npz", **all_maps)
    _plot_bars(
        all_model_rows,
        "mean_avf",
        "std_avf",
        RESULTS_DIR / "fig6.png",
        "Model AVF (%)",
    )
    _plot_bars(
        all_util_rows,
        "mean_utilization",
        "std_utilization",
        RESULTS_DIR / "fig7.png",
        "PE utilization (%)",
    )
    _plot_layers(all_layer_rows, RESULTS_DIR / "fig12.png")
    _plot_maps(all_maps, RESULTS_DIR / "fig11.png")
    metadata["wall_clock_seconds"] = time.perf_counter() - started
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--images", type=int, default=100)
    parser.add_argument("--vgg-images", type=int, default=20)
    args = parser.parse_args()
    metadata = run_sweep(
        tuple(args.models),
        image_count=args.images,
        vgg_image_count=max(20, args.vgg_images),
    )
    print(metadata)


if __name__ == "__main__":
    main()
