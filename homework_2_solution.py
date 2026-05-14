from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


PRACTICE_DIR = Path(__file__).parent / "practice_2"
OUTPUT_DIR = Path(__file__).parent / "homework_2_outputs"
if str(PRACTICE_DIR) not in sys.path:
    sys.path.insert(0, str(PRACTICE_DIR))

from world import World
from agent import Agent
from saliency import SaliencyMap
from model import WorldModel
from action import NearestUnknownPolicy


def _to_cycle(feature):
    if isinstance(feature, list):
        return [frame.astype(np.uint8) for frame in feature]
    return [feature.astype(np.uint8)]


def _pad_frame(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    y0 = (shape[0] - frame.shape[0]) // 2
    x0 = (shape[1] - frame.shape[1]) // 2
    out[y0:y0 + frame.shape[0], x0:x0 + frame.shape[1]] = frame
    return out


def phase_invariant_hamming(feature_a, feature_b) -> float:
    cycle_a = _to_cycle(feature_a)
    cycle_b = _to_cycle(feature_b)
    if len(cycle_a) != len(cycle_b):
        return float("inf")

    h = max(max(frame.shape[0] for frame in cycle_a), max(frame.shape[0] for frame in cycle_b))
    w = max(max(frame.shape[1] for frame in cycle_a), max(frame.shape[1] for frame in cycle_b))
    cycle_a = [_pad_frame(frame, (h, w)) for frame in cycle_a]
    cycle_b = [_pad_frame(frame, (h, w)) for frame in cycle_b]

    best = float("inf")
    period = len(cycle_a)
    for shift in range(period):
        score = 0.0
        for i in range(period):
            score += np.abs(cycle_a[i].astype(np.int16) - cycle_b[(i + shift) % period].astype(np.int16)).sum()
        best = min(best, score)
    return best


def group_objects(model: WorldModel, similarity_threshold: float = 0.0) -> list[dict]:
    groups = []
    for obj in model.objects.values():
        history = obj.history
        if history is None or history.period is None or history.cycle is None:
            continue

        period = history.period
        feature = history.cycle[0].copy() if period == 1 else [frame.copy() for frame in history.cycle]

        match = None
        for group in groups:
            if group["period"] != period:
                continue
            if phase_invariant_hamming(group["feature"], feature) <= similarity_threshold:
                match = group
                break

        if match is None:
            groups.append({"period": period, "feature": feature, "objects": [obj]})
        else:
            match["objects"].append(obj)

    return groups


def build_object_model(model: WorldModel, similarity_threshold: float = 0.0, grouped: bool = True) -> list[dict]:
    if grouped:
        groups = group_objects(model, similarity_threshold)
    else:
        groups = []
        for obj in model.objects.values():
            history = obj.history
            if history is None or history.period is None or history.cycle is None:
                continue
            period = history.period
            feature = history.cycle[0].copy() if period == 1 else [frame.copy() for frame in history.cycle]
            groups.append({"period": period, "feature": feature, "objects": [obj]})

    result = []
    for group in groups:
        positions = [(float(obj.centroid[0]), float(obj.centroid[1])) for obj in group["objects"]]
        result.append(
            {
                "feature": group["feature"],
                "period": group["period"],
                "positions": positions,
                "count": len(positions),
                "object_ids": [obj.id for obj in group["objects"]],
            }
        )
    return result


def _stamp(grid: np.ndarray, frame: np.ndarray, cx: float, cy: float) -> None:
    h, w = frame.shape
    x0 = int(round(cx)) - w // 2
    y0 = int(round(cy)) - h // 2
    ys = np.arange(y0, y0 + h) % grid.shape[0]
    xs = np.arange(x0, x0 + w) % grid.shape[1]
    grid[np.ix_(ys, xs)] = np.maximum(grid[np.ix_(ys, xs)], frame.astype(np.uint8))


def reconstruct_from_objects(object_model: list[dict], width: int, height: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.uint8)
    for cls in object_model:
        feature = cls["feature"][0] if isinstance(cls["feature"], list) else cls["feature"]
        for cx, cy in cls["positions"]:
            _stamp(grid, feature, cx, cy)
    return grid


def hamming_error(real_world: np.ndarray, reconstructed: np.ndarray) -> float:
    return np.abs(real_world.astype(np.int16) - reconstructed.astype(np.int16)).mean()


def model_size_bits(object_model: list[dict], width: int, height: int) -> int:
    coord_bits = 2 * int(np.ceil(np.log2(max(width, height))))
    total = 0
    for cls in object_model:
        feature = cls["feature"]
        feat_bits = sum(frame.size for frame in feature) if isinstance(feature, list) else feature.size
        total += feat_bits + cls["count"] * coord_bits
    return int(total)


def summarize_model(world: World, model: WorldModel, threshold: float) -> dict:
    grouped_model = build_object_model(model, similarity_threshold=threshold, grouped=True)
    ungrouped_model = build_object_model(model, grouped=False)

    grouped_grid = reconstruct_from_objects(grouped_model, world.width, world.height)
    ungrouped_grid = reconstruct_from_objects(ungrouped_model, world.width, world.height)

    return {
        "grouped_model": grouped_model,
        "ungrouped_model": ungrouped_model,
        "grouped_error": hamming_error(world.state, grouped_grid),
        "ungrouped_error": hamming_error(world.state, ungrouped_grid),
        "grouped_size_bits": model_size_bits(grouped_model, world.width, world.height),
        "ungrouped_size_bits": model_size_bits(ungrouped_model, world.width, world.height),
        "grouped_classes": len(grouped_model),
        "tracked_objects": len(model.objects),
    }


def _metric_row(world: World, model: WorldModel, threshold: float) -> dict:
    summary = summarize_model(world, model, threshold)
    return {
        "time_step": world.time_step,
        "coverage": model.coverage,
        "tracked_objects": summary["tracked_objects"],
        "grouped_classes": summary["grouped_classes"],
        "grouped_error": summary["grouped_error"],
        "ungrouped_error": summary["ungrouped_error"],
        "grouped_size_bits": summary["grouped_size_bits"],
        "ungrouped_size_bits": summary["ungrouped_size_bits"],
    }


def run_exploration(world_size=(50, 50), perception_radius=9, warmup=1000, max_steps=4000, threshold: float = 0.0,
                    sample_every: int = 25):
    np.random.seed(42)
    world = World(*world_size)
    for _ in range(warmup):
        world.step()

    start_pos = (world_size[0] // 2, world_size[1] // 2)
    agent = Agent(world, start_pos, perception_radius)
    saliency = SaliencyMap(2 * perception_radius + 1)
    model = WorldModel(*world_size, saliency=saliency)
    policy = NearestUnknownPolicy()
    timeline = []

    for step_idx in range(max_steps):
        world.step()
        model.update(agent.perceive(), agent.position, agent.perception_radius, world.time_step)

        if step_idx % sample_every == 0 or step_idx == max_steps - 1:
            timeline.append(_metric_row(world, model, threshold))

        direction = policy(agent, model)
        if direction is None:
            break
        agent.move(direction)

    if not timeline or timeline[-1]["time_step"] != world.time_step:
        timeline.append(_metric_row(world, model, threshold))

    return world, model, timeline


def run_threshold_sweep(model: WorldModel, world: World, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        summary = summarize_model(world, model, threshold)
        rows.append(
            {
                "threshold": threshold,
                "grouped_classes": summary["grouped_classes"],
                "grouped_error": summary["grouped_error"],
                "grouped_size_bits": summary["grouped_size_bits"],
            }
        )
    return rows


def save_timeline_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_threshold_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_error_over_time(rows: list[dict], path: Path) -> None:
    x = [row["time_step"] for row in rows]
    y_grouped = [row["grouped_error"] for row in rows]
    y_ungrouped = [row["ungrouped_error"] for row in rows]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y_grouped, label="Grouped object model", linewidth=2)
    plt.plot(x, y_ungrouped, label="Ungrouped object model", linewidth=2)
    plt.xlabel("World time step")
    plt.ylabel("Hamming error")
    plt.title("Reconstruction Error Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_threshold_tradeoff(rows: list[dict], path: Path) -> None:
    x = [row["threshold"] for row in rows]
    y_err = [row["grouped_error"] for row in rows]
    y_size = [row["grouped_size_bits"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(x, y_err, color="tab:red", marker="o", linewidth=2)
    ax1.set_xlabel("Grouping threshold")
    ax1.set_ylabel("Hamming error", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, y_size, color="tab:blue", marker="s", linewidth=2)
    ax2.set_ylabel("Model size (bits)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Threshold Tradeoff: Error vs Size")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_model_size_comparison(raster_bits: int, ungrouped_bits: int, grouped_bits: int, path: Path) -> None:
    labels = ["Raster", "Object ungrouped", "Object grouped"]
    values = [raster_bits, ungrouped_bits, grouped_bits]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values, color=["#666666", "#4c78a8", "#72b7b2"])
    plt.ylabel("Bits")
    plt.title("Model Size Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_table(world: World, model: WorldModel, summary: dict, path: Path) -> None:
    raster_bits = world.width * world.height
    grouped_error = summary["grouped_error"]
    ungrouped_error = summary["ungrouped_error"]

    lines = [
        "| Model | Size (bits) | Hamming error |",
        "|---|---:|---:|",
        f"| Raster map | {raster_bits} | 0.000000 |",
        f"| Object model (ungrouped) | {summary['ungrouped_size_bits']} | {ungrouped_error:.6f} |",
        f"| Object model (grouped) | {summary['grouped_size_bits']} | {grouped_error:.6f} |",
    ]
    path.write_text("\n".join(lines) + "\n")


def save_report(world: World, model: WorldModel, summary: dict, threshold_rows: list[dict], path: Path) -> None:
    grouped_model = summary["grouped_model"]
    best_threshold = min(threshold_rows, key=lambda row: (row["grouped_error"], row["grouped_size_bits"]))

    lines = [
        "# Homework 2 Report",
        "",
        "## Found Object Classes",
        "",
    ]

    for i, cls in enumerate(grouped_model, start=1):
        feature = cls["feature"]
        shape = feature[0].shape if isinstance(feature, list) else feature.shape
        positions = ", ".join(f"({x:.1f}, {y:.1f})" for x, y in cls["positions"])
        lines.append(
            f"- Class {i}: period={cls['period']}, count={cls['count']}, "
            f"feature_shape={shape}, positions={positions}"
        )

    lines += [
        "",
        "## Model Comparison",
        "",
        f"- World size: {world.width}x{world.height}",
        f"- Coverage: {model.coverage:.4f}",
        f"- Tracked objects: {len(model.objects)}",
        f"- Grouped classes: {summary['grouped_classes']}",
        f"- Raster size: {world.width * world.height} bits",
        f"- Object size without grouping: {summary['ungrouped_size_bits']} bits",
        f"- Object size with grouping: {summary['grouped_size_bits']} bits",
        "",
        "## Accuracy vs Compression",
        "",
        f"- Hamming error without grouping: {summary['ungrouped_error']:.6f}",
        f"- Hamming error with grouping: {summary['grouped_error']:.6f}",
        f"- Best threshold from sweep: {best_threshold['threshold']}",
        f"- Error at best threshold: {best_threshold['grouped_error']:.6f}",
        f"- Size at best threshold: {best_threshold['grouped_size_bits']} bits",
        "",
        "## Conclusions",
        "",
        "- The grouped object model is smaller than the raster map and also smaller than the ungrouped object model.",
        "- Reconstruction is lossy because the model stores detected live-object structure, not the full background and not every segmentation detail.",
        "- Lower grouping thresholds preserve more distinct classes, while larger thresholds compress more aggressively and can merge different objects.",
        "- This solution does not model dynamic object motion prediction; it focuses on the required non-optional parts.",
        "",
        "## Files",
        "",
        "- `homework_2_outputs/error_over_time.png`",
        "- `homework_2_outputs/threshold_tradeoff.png`",
        "- `homework_2_outputs/model_size_comparison.png`",
        "- `homework_2_outputs/model_comparison_table.md`",
        "- `homework_2_outputs/timeline_metrics.csv`",
        "- `homework_2_outputs/threshold_sweep.csv`",
    ]

    path.write_text("\n".join(lines) + "\n")


def print_summary(world: World, model: WorldModel, summary: dict) -> None:
    print(f"world_size={world.width}x{world.height}")
    print(f"time_step={world.time_step}")
    print(f"coverage={model.coverage:.4f}")
    print(f"tracked_objects={len(model.objects)}")
    print(f"grouped_classes={summary['grouped_classes']}")
    print(f"grouped_instances={sum(cls['count'] for cls in summary['grouped_model'])}")
    print(f"ungrouped_instances={sum(cls['count'] for cls in summary['ungrouped_model'])}")
    print(f"raster_size_bits={world.width * world.height}")
    print(f"object_size_bits_grouped={summary['grouped_size_bits']}")
    print(f"object_size_bits_ungrouped={summary['ungrouped_size_bits']}")
    print(f"hamming_error_grouped={summary['grouped_error']:.6f}")
    print(f"hamming_error_ungrouped={summary['ungrouped_error']:.6f}")

    for i, cls in enumerate(summary["grouped_model"], start=1):
        feature = cls["feature"]
        shape = feature[0].shape if isinstance(feature, list) else feature.shape
        positions = [(round(x, 1), round(y, 1)) for x, y in cls["positions"]]
        print(f"class_{i}: period={cls['period']} count={cls['count']} feature_shape={shape} positions={positions}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--radius", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0, 20, 40, 80, 120, 200])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    world, model, timeline = run_exploration(
        world_size=(args.width, args.height),
        perception_radius=args.radius,
        warmup=args.warmup,
        max_steps=args.max_steps,
        threshold=args.threshold,
        sample_every=args.sample_every,
    )
    summary = summarize_model(world, model, args.threshold)
    threshold_rows = run_threshold_sweep(model, world, args.thresholds)

    save_timeline_csv(timeline, OUTPUT_DIR / "timeline_metrics.csv")
    save_threshold_csv(threshold_rows, OUTPUT_DIR / "threshold_sweep.csv")
    plot_error_over_time(timeline, OUTPUT_DIR / "error_over_time.png")
    plot_threshold_tradeoff(threshold_rows, OUTPUT_DIR / "threshold_tradeoff.png")
    plot_model_size_comparison(
        world.width * world.height,
        summary["ungrouped_size_bits"],
        summary["grouped_size_bits"],
        OUTPUT_DIR / "model_size_comparison.png",
    )
    save_table(world, model, summary, OUTPUT_DIR / "model_comparison_table.md")
    save_report(world, model, summary, threshold_rows, OUTPUT_DIR / "report.md")

    print_summary(world, model, summary)
    print(f"outputs_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
