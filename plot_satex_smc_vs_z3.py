#!/usr/bin/env python3
"""Plot SMC and Z3 runtime by SatEX test case from the comparison CSV."""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
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
        description="Plot SMC and Z3 runtimes by SatEX test case from satex_smc_z3_results.csv."
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
        help="Use a linear runtime axis instead of a logarithmic runtime axis.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide the legend.",
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


def test_case_number(benchmark: str) -> int | None:
    stem = Path(benchmark).stem
    match = re.fullmatch(r"(?:test[_-]?case[_-]?)?(\d+)", stem, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def normalize_group(group: str, benchmark: str) -> str:
    parts = Path(group.strip()).parts if group.strip() else Path(benchmark).parts[:-1]
    if parts and parts[-1] == "z3_smtlib":
        parts = parts[:-1]
    return str(Path(*parts)) if parts else "unknown"


def runtime_points(
    rows: list[dict[str, str]], timeout_sec: float | None
) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for row in rows:
        solver = row["solver"].strip().lower()
        if solver not in {"smc", "z3"}:
            continue

        test_case = test_case_number(row["benchmark"])
        if test_case is None or test_case <= 0:
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
                "group": normalize_group(row.get("group", ""), row["benchmark"]),
                "solver": solver.upper() if solver == "z3" else "SMC",
                "test_case_number": test_case,
                "runtime": runtime,
                "status": row["status"],
            }
        )
    return points


def line_series(
    points: list[dict[str, object]],
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    runtimes: dict[tuple[str, str, int], list[float]] = {}
    for point in points:
        key = (
            str(point["group"]),
            str(point["solver"]),
            int(point["test_case_number"]),
        )
        runtimes.setdefault(key, []).append(float(point["runtime"]))

    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for (group, solver, test_case), values in runtimes.items():
        series.setdefault((group, solver), []).append(
            (
                test_case,
                statistics.median(values),
            )
        )

    return {
        key: sorted(values, key=lambda item: item[0])
        for key, values in series.items()
    }


def plot(
    points: list[dict[str, object]],
    output_path: Path,
    linear_axes: bool,
    show_legend: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/smc-matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.5,
        }
    )

    groups = sorted({str(point["group"]) for point in points})
    group_colors = {
        "Scalability Boolean1": "#1f4de3",
        "Scalability Boolean2": "#111111",
        "Scalability Real": "#8a5a2b",
        "Scalability UNSAT": "#d62728",
    }
    fallback_colors = plt.get_cmap("tab10")
    colors = {
        group: group_colors.get(group, fallback_colors(index % 10))
        for index, group in enumerate(groups)
    }
    markers = {
        "Scalability Boolean1": "o",
        "Scalability Boolean2": "x",
        "Scalability Real": "o",
        "Scalability UNSAT": "s",
    }
    linestyles = {"SMC": "-", "Z3": "--"}
    series = line_series(points)

    fig, ax = plt.subplots(figsize=(4.2, 1.65))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.26, top=0.70)
    for group in groups:
        for solver in ("SMC", "Z3"):
            values = series.get((group, solver), [])
            if not values:
                continue

            x_values = [test_case for test_case, _runtime in values]
            y_values = [runtime for _test_case, runtime in values]
            ax.plot(
                x_values,
                y_values,
                label=f"{group} {solver}",
                color=colors[group],
                linestyle=linestyles[solver],
                marker=markers.get(group, "o"),
                markersize=2.2,
                markeredgewidth=0.5,
                linewidth=0.55,
            )

    all_x_values = [
        test_case
        for values in series.values()
        for test_case, _runtime in values
    ]
    all_runtimes = [
        runtime
        for values in series.values()
        for _test_case, runtime in values
    ]
    min_x = min(all_x_values)
    max_x = max(all_x_values)
    min_runtime = min(all_runtimes)
    max_runtime = max(all_runtimes)

    if linear_axes:
        runtime_padding = 0.05 * (max_runtime - min_runtime)
        y_min = max(0.0, min_runtime - runtime_padding)
        y_max = max_runtime + runtime_padding
    else:
        y_min = min_runtime / 1.25
        y_max = max_runtime * 1.25
        ax.set_yscale("log")

    ax.set_xlim(min_x, max_x + 0.5)
    ax.set_ylim(y_min, y_max)
    tick_step = max(1, round((max_x - min_x) / 8))
    x_ticks = list(range(min_x, max_x + 1, tick_step))
    if x_ticks[-1] != max_x:
        x_ticks.append(max_x)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Test Case Number")
    ax.set_ylabel("Runtime (seconds)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.25, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")
    ax.tick_params(width=0.7, length=3)
    if show_legend:
        ax.legend(
            ncols=2,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            frameon=False,
            borderaxespad=0.0,
        )

    fig.savefig(output_path, dpi=300)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    points = runtime_points(rows, args.timeout_sec)
    if not points:
        raise ValueError(
            "No SMC/Z3 benchmark rows with a parseable test case number found in the CSV."
        )

    plot(points, args.output, args.linear, not args.no_legend)
    print(f"Plotted {len(points)} SMC/Z3 runtime runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
