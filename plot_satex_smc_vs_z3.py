#!/usr/bin/env python3
"""Plot SMC and Z3 runtime by SatEX formula size from the comparison CSV."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


def default_input_path(repo_root: Path) -> Path:
    candidates = (
        repo_root / "result" / "satex_smc_z3_results.csv",
        repo_root / "results" / "satex_smc_z3_results.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot SMC and Z3 runtimes by formula size from satex_smc_z3_results.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(repo_root),
        help="CSV file produced by compare_satex_smc_z3.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "satex_smc_vs_z3.png",
        help="Path for the generated figure.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=positive_float,
        default=None,
        help="Optional value to use for timed-out runs instead of their measured elapsed time.",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear axes instead of logarithmic axes.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    required = {"solver", "benchmark", "group", "run", "status", "elapsed_sec"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required column(s): {missing_list}")
    return rows


def formula_size(benchmark: str) -> int | None:
    stem = Path(benchmark).stem
    match = re.search(r"(?:constraints|real_vars|sensors)_(\d+)", stem)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+)", stem)
    if match:
        return int(match.group(1))

    return None


def runtime_points(
    rows: list[dict[str, str]], timeout_sec: float | None
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for row in rows:
        solver = row["solver"].strip().lower()
        if solver not in {"smc", "z3"}:
            continue

        size = formula_size(row["benchmark"])
        if size is None or size <= 0:
            continue

        runtime = (
            timeout_sec
            if row["status"] == "timeout" and timeout_sec
            else float(row["elapsed_sec"])
        )
        if runtime <= 0:
            continue

        points.append(
            {
                "benchmark": row["benchmark"],
                "run": row["run"],
                "group": row.get("group") or "unknown",
                "solver": solver.upper() if solver == "z3" else "SMC",
                "formula_size": size,
                "runtime": runtime,
                "status": row["status"],
            }
        )
    return points


def plot(points: list[dict[str, object]], output_path: Path, linear_axes: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/smc-matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    groups = sorted({str(point["group"]) for point in points})
    colormap = plt.get_cmap("tab10")
    colors = {group: colormap(index % 10) for index, group in enumerate(groups)}
    markers = {"SMC": "o", "Z3": "^"}

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for group in groups:
        for solver in ("SMC", "Z3"):
            solver_points = [
                point
                for point in points
                if point["group"] == group and point["solver"] == solver
            ]
            if not solver_points:
                continue

            ax.scatter(
                [int(point["formula_size"]) for point in solver_points],
                [float(point["runtime"]) for point in solver_points],
                label=f"{group} - {solver}",
                alpha=0.78,
                s=46,
                marker=markers[solver],
                edgecolors="white",
                linewidths=0.4,
                color=colors[group],
            )

    all_sizes = [int(point["formula_size"]) for point in points]
    all_runtimes = [float(point["runtime"]) for point in points]
    min_size = min(all_sizes)
    max_size = max(all_sizes)
    min_runtime = min(all_runtimes)
    max_runtime = max(all_runtimes)

    if linear_axes:
        size_padding = 0.05 * (max_size - min_size)
        runtime_padding = 0.05 * (max_runtime - min_runtime)
        x_min = max(0.0, min_size - size_padding)
        x_max = max_size + size_padding
        y_min = max(0.0, min_runtime - runtime_padding)
        y_max = max_runtime + runtime_padding
    else:
        x_min = min_size / 1.25
        x_max = max_size * 1.25
        y_min = min_runtime / 1.25
        y_max = max_runtime * 1.25
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Formula size")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("SatEX Runtime by Formula Size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
    ax.legend(title="Benchmark group / solver", fontsize=8)

    fig.savefig(output_path, dpi=300)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    points = runtime_points(rows, args.timeout_sec)
    if not points:
        raise ValueError(
            "No SMC/Z3 benchmark rows with a parseable formula size found in the CSV."
        )

    plot(points, args.output, args.linear)
    print(f"Plotted {len(points)} SMC/Z3 runtime runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
