#!/usr/bin/env python3
"""Compare bin/SMC and Z3 on all SatEX SMT-LIB benchmarks.

The script records one row per solver/file/run. It intentionally treats
timeouts and non-zero exits as data so one failing benchmark does not stop the
whole experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class RunResult:
    solver: str
    benchmark: str
    group: str
    run: int
    status: str
    result: str
    elapsed_sec: float
    returncode: int | None
    timed_out: bool


def default_dataset_root(repo_root: Path) -> Path:
    for candidate in (repo_root / "data" / "SatEX", repo_root / "data" / "Satex"):
        if candidate.exists():
            return candidate
    return repo_root / "data" / "SatEX"


def default_smc_binary(repo_root: Path) -> Path:
    for candidate in (repo_root / "bin" / "SMC", repo_root / "bin" / "smc"):
        if candidate.exists():
            return candidate
    return repo_root / "bin" / "SMC"


def discover_benchmarks(dataset_root: Path) -> list[Path]:
    benchmarks = sorted(dataset_root.rglob("*.smt2"), key=lambda path: str(path))
    if not benchmarks:
        raise FileNotFoundError(f"No .smt2 files found under {dataset_root}")
    return benchmarks


def run_solver(
    solver_name: str,
    command: list[str],
    benchmark: Path,
    dataset_root: Path,
    run_index: int,
    timeout_sec: float,
) -> RunResult:
    start = time.perf_counter()
    timed_out = False
    returncode: int | None = None

    try:
        completed = subprocess.run(
            [*command, str(benchmark)],
            check=False,
            timeout=timeout_sec,
        )
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    elapsed_sec = time.perf_counter() - start

    if timed_out:
        status = "timeout"
    elif returncode == 0:
        status = "ok"
    else:
        status = "error"

    rel_benchmark = benchmark.relative_to(dataset_root)
    return RunResult(
        solver=solver_name,
        benchmark=str(rel_benchmark),
        group=str(rel_benchmark.parent),
        run=run_index,
        status=status,
        result="",
        elapsed_sec=elapsed_sec,
        returncode=returncode,
        timed_out=timed_out,
    )


def write_csv(path: Path, rows: Iterable[RunResult]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, rows: Iterable[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as json_file:
        json.dump([asdict(row) for row in rows], json_file, indent=2)
        json_file.write("\n")


def summarize(rows: list[RunResult]) -> dict[str, object]:
    by_solver: dict[str, list[RunResult]] = {}
    for row in rows:
        by_solver.setdefault(row.solver, []).append(row)

    solver_summary = {}
    for solver, solver_rows in by_solver.items():
        ok_times = [row.elapsed_sec for row in solver_rows if row.status == "ok"]
        solver_summary[solver] = {
            "runs": len(solver_rows),
            "ok": sum(row.status == "ok" for row in solver_rows),
            "timeout": sum(row.status == "timeout" for row in solver_rows),
            "error": sum(row.status == "error" for row in solver_rows),
            "median_ok_sec": statistics.median(ok_times) if ok_times else None,
            "total_sec": sum(row.elapsed_sec for row in solver_rows),
        }

    paired: dict[tuple[str, int], dict[str, RunResult]] = {}
    for row in rows:
        paired.setdefault((row.benchmark, row.run), {})[row.solver] = row

    disagreements = []
    for (benchmark, run_index), solver_rows in paired.items():
        smc = solver_rows.get("smc")
        z3 = solver_rows.get("z3")
        if not smc or not z3:
            continue
        if smc.status == "ok" and z3.status == "ok" and smc.result != z3.result:
            disagreements.append(
                {
                    "benchmark": benchmark,
                    "run": run_index,
                    "smc": smc.result,
                    "z3": z3.result,
                }
            )

    return {
        "solver_summary": solver_summary,
        "disagreements": disagreements,
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare bin/SMC and Z3 on every .smt2 file under data/SatEX."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(repo_root),
        help="Directory containing SatEX .smt2 benchmarks.",
    )
    parser.add_argument(
        "--smc",
        type=Path,
        default=default_smc_binary(repo_root),
        help="Path to the SMC executable.",
    )
    parser.add_argument(
        "--z3",
        default="z3",
        help="Z3 executable name or path.",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=300.0,
        help="Per solver per benchmark timeout in seconds.",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=1,
        help="Number of runs for each solver/benchmark pair.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repo_root / "results" / "satex_smc_z3_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="Run only the first N benchmarks after sorting. Useful for smoke tests.",
    )
    parser.add_argument(
        "--z3-extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass to Z3 before the benchmark path. Repeatable.",
    )
    parser.add_argument(
        "--smc-extra-arg",
        action="append",
        default=[],
        help="Extra argument to pass to SMC before the benchmark path. Repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    smc_binary = args.smc.resolve()

    if not dataset_root.exists():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not smc_binary.exists():
        print(f"SMC executable does not exist: {smc_binary}", file=sys.stderr)
        return 2

    benchmarks = discover_benchmarks(dataset_root)
    if args.limit is not None:
        benchmarks = benchmarks[: args.limit]

    solvers = [
        ("smc", [str(smc_binary), *args.smc_extra_arg]),
        ("z3", [args.z3, *args.z3_extra_arg]),
    ]

    total_runs = len(benchmarks) * len(solvers) * args.repetitions
    print(f"Dataset: {dataset_root}")
    print(f"Benchmarks: {len(benchmarks)}")
    print(f"Total solver runs: {total_runs}")

    rows: list[RunResult] = []
    completed_runs = 0
    for run_index in range(1, args.repetitions + 1):
        for benchmark in benchmarks:
            for solver_name, command in solvers:
                completed_runs += 1
                rel = benchmark.relative_to(dataset_root)
                print(
                    f"[{completed_runs}/{total_runs}] {solver_name} run "
                    f"{run_index}: {rel}",
                    flush=True,
                )
                rows.append(
                    run_solver(
                        solver_name=solver_name,
                        command=command,
                        benchmark=benchmark,
                        dataset_root=dataset_root,
                        run_index=run_index,
                        timeout_sec=args.timeout,
                    )
                )

    write_csv(args.output_csv, rows)
    if args.output_json:
        write_json(args.output_json, rows)

    summary = summarize(rows)
    print(f"Wrote CSV: {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON: {args.output_json}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
