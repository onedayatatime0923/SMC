#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import scipy as sci
import z3


ROOT = Path(__file__).resolve().parent


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def bool_var_to_int(b):
    return z3.If(b, z3.IntVal(1), z3.IntVal(0))


def read_cnf_constraints(path):
    with path.open() as fh:
        content = fh.readlines()

    document_start = 0
    for line in content:
        if line[0] == "c":
            document_start += 1
        else:
            break

    problem_def_line = content[document_start].split(" ")
    number_of_bool_vars = int(problem_def_line[2])
    max_number_of_constraints = int(problem_def_line[3])
    clauses = []
    for line in content[document_start + 1 : document_start + 1 + max_number_of_constraints]:
        parts = line.split(" ")[0:-1]
        clauses.append([int(item) for item in parts])
    return number_of_bool_vars, clauses


def add_boolean_clause(solver, b_vars, clause):
    positive_vars = [abs(i) - 1 for i in clause if i > 0]
    negative_vars = [abs(i) - 1 for i in clause if i < 0]
    z3_clause = [b_vars[i] for i in positive_vars] + [z3.Not(b_vars[i]) for i in negative_vars]
    solver.add(z3.Or(*z3_clause))


def format_manifest(manifest_path, entries):
    with manifest_path.open("w") as fh:
        json.dump(entries, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_solver(path, solver):
    with path.open("w") as fh:
        fh.write(solver.to_smt2())


def generate_boolean_family(example_dir, number_of_real_vars, test_cases):
    np.random.seed(0)
    output_dir = example_dir / "z3_smtlib"
    ensure_dir(output_dir)

    number_of_bool_vars, clauses = read_cnf_constraints(example_dir / "003-23-80.cnf")
    manifest = []

    for number_of_constraints in test_cases:
        prob_convex = 1.0
        convex_choice = np.random.choice([0, 1], number_of_constraints, p=[1 - prob_convex, prob_convex])
        number_of_convex_constraints = int(number_of_real_vars / 2)
        a_matrix = sci.sparse.rand(number_of_convex_constraints, number_of_real_vars, density=0.9)
        a_matrix = 10 * a_matrix - a_matrix.ceil()
        a_matrix = a_matrix.toarray()
        b_vector = sci.sparse.rand(number_of_convex_constraints, 1, density=1.0)
        b_vector = 100 * b_vector - b_vector.ceil()
        b_vector = b_vector.toarray()

        solver = z3.Solver()
        b_vars = z3.BoolVector("b", number_of_bool_vars)
        r_vars = z3.RealVector("y", number_of_real_vars)

        convex_constraint_counter = 0
        for counter in range(number_of_constraints):
            bool_constraint = clauses[counter]
            add_boolean_clause(solver, b_vars, bool_constraint)

            if convex_constraint_counter < number_of_convex_constraints and convex_choice[counter] == 1:
                first_var = abs(bool_constraint[-1]) - 1
                coeffs = a_matrix[convex_constraint_counter, :]
                rhs = float(b_vector[convex_constraint_counter][0])
                lin_const = z3.Sum([r_vars[i] * coeffs[i] for i in range(number_of_real_vars)]) - rhs
                solver.add(z3.Implies(b_vars[first_var], lin_const < 0))
                convex_constraint_counter += 1

        output_path = output_dir / f"constraints_{number_of_constraints}.smt2"
        if not output_path.exists():
            write_solver(output_path, solver)
        manifest.append(
            {
                "test_case": number_of_constraints,
                "output": str(output_path.relative_to(example_dir)),
                "real_vars": number_of_real_vars,
                "bool_vars": number_of_bool_vars,
                "bool_constraints": number_of_constraints,
            }
        )

    return manifest


def generate_real_family(example_dir):
    np.random.seed(0)
    output_dir = example_dir / "z3_smtlib"
    ensure_dir(output_dir)

    number_of_bool_vars, clauses = read_cnf_constraints(example_dir / "003-23-80.cnf")
    number_of_boolean_constraints = 7000
    number_of_real_vars_test_cases = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    manifest = []

    for number_of_real_vars in number_of_real_vars_test_cases:
        number_of_constraints = number_of_boolean_constraints
        prob_convex = 1.0
        convex_choice = np.random.choice([0, 1], number_of_constraints, p=[1 - prob_convex, prob_convex])
        number_of_convex_constraints = int(number_of_real_vars / 2)
        a_matrix = sci.sparse.rand(number_of_convex_constraints, number_of_real_vars, density=0.9)
        a_matrix = 10 * a_matrix - a_matrix.ceil()
        a_matrix = a_matrix.toarray()
        b_vector = sci.sparse.rand(number_of_convex_constraints, 1, density=1.0)
        b_vector = 100 * b_vector - b_vector.ceil()
        b_vector = b_vector.toarray()

        solver = z3.Solver()
        b_vars = z3.BoolVector("b", number_of_bool_vars)
        r_vars = z3.RealVector("y", number_of_real_vars)

        convex_constraint_counter = 0
        for counter in range(number_of_constraints):
            bool_constraint = clauses[counter]
            add_boolean_clause(solver, b_vars, bool_constraint)

            if convex_constraint_counter < number_of_convex_constraints and convex_choice[counter] == 1:
                first_var = abs(bool_constraint[-1]) - 1
                coeffs = a_matrix[convex_constraint_counter, :]
                rhs = float(b_vector[convex_constraint_counter][0])
                lin_const = z3.Sum([r_vars[i] * coeffs[i] for i in range(number_of_real_vars)]) - rhs
                solver.add(z3.Implies(b_vars[first_var], lin_const < 0))
                convex_constraint_counter += 1

        output_path = output_dir / f"real_vars_{number_of_real_vars}.smt2"
        if not output_path.exists():
            write_solver(output_path, solver)
        manifest.append(
            {
                "test_case": number_of_real_vars,
                "output": str(output_path.relative_to(example_dir)),
                "real_vars": number_of_real_vars,
                "bool_vars": number_of_bool_vars,
                "bool_constraints": number_of_constraints,
            }
        )

    return manifest


def obsv(a_matrix, c_row):
    n = a_matrix.shape[0]
    blocks = []
    current = np.array(c_row, dtype=float)
    for _ in range(n):
        blocks.append(np.array(current, dtype=float))
        current = np.dot(current, a_matrix)
    return np.vstack(blocks)


def sse_random(n, p, s_bar, attack_power):
    np.random.seed(0)

    a_matrix = sci.sparse.rand(n, n, density=1.0).toarray()
    c_matrix = (10 * sci.sparse.rand(p, n, density=1.0)).toarray()

    _, singular_values, _ = np.linalg.svd(a_matrix)
    max_eig = max(singular_values)
    a_matrix = a_matrix / (max_eig + 0.1)

    attacked_sensor_index = np.random.permutation(p - 1)[0:s_bar]
    x_state = np.random.rand(n, 1)
    y_matrix = np.transpose(np.dot(c_matrix, x_state))

    for _ in range(0, n - 1):
        x_state = np.dot(a_matrix, x_state)
        attack_signal = np.zeros((p, 1))
        attack_signal[attacked_sensor_index] = attack_power * np.random.rand(len(attacked_sensor_index), 1)
        y_sample = np.dot(c_matrix, x_state) + attack_signal
        y_matrix = np.append(y_matrix, np.transpose(y_sample), axis=0)

    return a_matrix, c_matrix, y_matrix


def generate_secure_state_estimation(example_dir):
    output_dir = example_dir / "z3_smtlib"
    ensure_dir(output_dir)

    max_sensors_under_attack = 5
    attack_power = 1.0
    n = 5
    number_of_sensors = [25, 50, 75, 100, 125, 150, 175, 200]
    manifest = []

    for p in number_of_sensors:
        a_matrix, c_matrix, y_matrix = sse_random(n, p, max_sensors_under_attack, attack_power)
        solver = z3.Solver()
        b_vars = z3.BoolVector("b", p)
        r_vars = z3.RealVector("y", n)

        for counter in range(p):
            o_i = obsv(a_matrix, c_matrix[counter, :])
            y_i = y_matrix[:, counter]
            q_i = np.dot(np.transpose(o_i), o_i)
            c_i = (-2 * np.dot(np.transpose(y_i), o_i)).reshape(-1)
            b_i = float(-1 * np.dot(np.transpose(y_i), y_i))

            quad_terms = []
            for row in range(n):
                for col in range(n):
                    quad_terms.append(r_vars[row] * r_vars[col] * q_i[row, col])
            quad_constraint = z3.Sum(quad_terms)
            linear_constraint = z3.Sum([r_vars[i] * c_i[i] for i in range(n)])
            solver.add(z3.Implies(b_vars[counter], quad_constraint + linear_constraint < b_i))

        solver.add(z3.Sum([bool_var_to_int(b_vars[i]) for i in range(p)]) == max_sensors_under_attack)

        output_path = output_dir / f"sensors_{p}.smt2"
        if not output_path.exists():
            write_solver(output_path, solver)
        manifest.append(
            {
                "test_case": p,
                "output": str(output_path.relative_to(example_dir)),
                "real_vars": n,
                "bool_vars": p,
                "bool_constraints": p,
            }
        )

    return manifest


def generate_unsat_family(example_dir):
    np.random.seed(0)
    output_dir = example_dir / "z3_smtlib"
    ensure_dir(output_dir)

    infiles = [
        "uuf225-086.cnf",
        "uuf225-019.cnf",
        "uuf225-026.cnf",
        "uuf225-0100.cnf",
        "uuf225-095.cnf",
        "uuf225-037.cnf",
        "uuf225-029.cnf",
        "uuf225-012.cnf",
        "uuf225-066.cnf",
    ]
    number_of_real_vars = 50
    manifest = []

    for infile in infiles:
        number_of_bool_vars, clauses = read_cnf_constraints(example_dir / infile)
        number_of_constraints = len(clauses)
        prob_convex = 1.0
        convex_choice = np.random.choice([0, 1], number_of_constraints, p=[1 - prob_convex, prob_convex])
        number_of_convex_constraints = int(number_of_real_vars / 2)
        a_matrix = sci.sparse.rand(number_of_convex_constraints, number_of_real_vars, density=0.9)
        a_matrix = 10 * a_matrix - a_matrix.ceil()
        a_matrix = a_matrix.toarray()
        b_vector = sci.sparse.rand(number_of_convex_constraints, 1, density=1.0)
        b_vector = 100 * b_vector - b_vector.ceil()
        b_vector = b_vector.toarray()

        solver = z3.Solver()
        b_vars = z3.BoolVector("b", number_of_bool_vars)
        r_vars = z3.RealVector("y", number_of_real_vars)

        convex_constraint_counter = 0
        for counter in range(number_of_constraints):
            bool_constraint = clauses[counter]
            add_boolean_clause(solver, b_vars, bool_constraint)

            if convex_constraint_counter < number_of_convex_constraints and convex_choice[counter] == 1:
                first_var = abs(bool_constraint[-1]) - 1
                coeffs = a_matrix[convex_constraint_counter, :]
                rhs = float(b_vector[convex_constraint_counter][0])
                lin_const = z3.Sum([r_vars[i] * coeffs[i] for i in range(number_of_real_vars)]) - rhs
                solver.add(z3.Implies(b_vars[first_var], lin_const < 0))
                convex_constraint_counter += 1

        stem = Path(infile).stem
        output_path = output_dir / f"real_vars_{number_of_real_vars}_{stem}.smt2"
        if not output_path.exists():
            write_solver(output_path, solver)
        manifest.append(
            {
                "test_case": infile,
                "output": str(output_path.relative_to(example_dir)),
                "real_vars": number_of_real_vars,
                "bool_vars": number_of_bool_vars,
                "bool_constraints": number_of_constraints,
            }
        )

    return manifest


BOOLEAN_TEST_CASES = [
    1000,
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    45000,
    50000,
    55000,
    60000,
    65000,
    70000,
    75000,
    80000,
    85000,
    90000,
    95000,
    100000,
    105000,
    110000,
    115000,
    120000,
    125000,
    130000,
]


def generate_target(target):
    if target == "boolean1":
        return "Scalability Boolean1", {
            "status": "generated",
            "files": generate_boolean_family(
                ROOT / "Scalability Boolean1",
                number_of_real_vars=100,
                test_cases=BOOLEAN_TEST_CASES,
            ),
        }
    if target == "boolean2":
        return "Scalability Boolean2", {
            "status": "generated",
            "files": generate_boolean_family(
                ROOT / "Scalability Boolean2",
                number_of_real_vars=1000,
                test_cases=BOOLEAN_TEST_CASES,
            ),
        }
    if target == "real":
        return "Scalability Real", {
            "status": "generated",
            "files": generate_real_family(ROOT / "Scalability Real"),
        }
    if target == "unsat":
        return "Scalability UNSAT", {
            "status": "generated",
            "files": generate_unsat_family(ROOT / "Scalability UNSAT"),
        }
    if target == "sse":
        return "Secure State Estimation", {
            "status": "generated",
            "files": generate_secure_state_estimation(ROOT / "Secure State Estimation"),
        }
    if target == "ltl":
        return "LTL Motion Planning", {
            "status": "skipped",
            "reason": "The case-study script does not define a Z3 benchmark or SMT encoding to export.",
        }
    raise ValueError("unknown target: %s" % target)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="*",
        choices=[
            "boolean1",
            "boolean2",
            "real",
            "unsat",
            "sse",
            "ltl",
            "all",
        ],
        default=["all"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if "all" in args.targets:
        ordered_targets = ["boolean1", "boolean2", "real", "unsat", "sse", "ltl"]
    else:
        ordered_targets = args.targets

    entries = {}
    for target in ordered_targets:
        key, value = generate_target(target)
        entries[key] = value

    format_manifest(ROOT / "z3_smtlib_generation_manifest.json", entries)


if __name__ == "__main__":
    main()
