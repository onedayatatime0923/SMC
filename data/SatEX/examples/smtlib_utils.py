import itertools
import math
import os


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def order_smt_outputs(paths):
    paths = [path for path in paths if path]
    ordered = sorted(paths, key=lambda path: (os.path.getsize(path), path))
    staged = []
    for index, path in enumerate(ordered, 1):
        staged_path = os.path.join(os.path.dirname(path), ".ordered_%d.smt2.tmp" % index)
        os.replace(path, staged_path)
        staged.append(staged_path)

    final_paths = []
    for index, path in enumerate(staged, 1):
        final_path = os.path.join(os.path.dirname(path), "%d.smt2" % index)
        os.replace(path, final_path)
        final_paths.append(final_path)
    return final_paths


def smt_real(value):
    try:
        value = float(value)
    except TypeError:
        value = float(value[0])

    if math.isinf(value) or math.isnan(value):
        raise ValueError("SMT-LIB real constants must be finite")

    if abs(value - round(value)) < 1e-12:
        text = str(int(round(value)))
    else:
        text = format(value, ".17g")

    if text.startswith("-"):
        return "(- %s)" % text[1:]
    return text


def smt_add(terms):
    terms = [term for term in terms if term != "0"]
    if not terms:
        return "0"
    if len(terms) == 1:
        return terms[0]
    return "(+ %s)" % " ".join(terms)


def smt_mul_real(coeff, var):
    coeff = float(coeff)
    if abs(coeff) < 1e-12:
        return "0"
    if abs(coeff - 1.0) < 1e-12:
        return var
    if abs(coeff + 1.0) < 1e-12:
        return "(- %s)" % var
    return "(* %s %s)" % (smt_real(coeff), var)


def smt_linear(coeffs, vars):
    return smt_add([smt_mul_real(coeff, var) for coeff, var in zip(coeffs, vars)])


def smt_linear_atom(coeffs, vars, rhs, sense):
    op = {"L": "<=", "G": ">=", "E": "="}.get(sense, "<=")
    return "(%s %s %s)" % (op, smt_linear(coeffs, vars), smt_real(rhs))


def smt_bool_lit(index, prefix="b"):
    name = "%s%d" % (prefix, abs(index) - 1)
    if index < 0:
        return "(not %s)" % name
    return name


def smt_or(literals):
    literals = list(literals)
    if not literals:
        return "false"
    if len(literals) == 1:
        return literals[0]
    return "(or %s)" % " ".join(literals)


def smt_and(literals):
    literals = list(literals)
    if not literals:
        return "true"
    if len(literals) == 1:
        return literals[0]
    return "(and %s)" % " ".join(literals)


def smt_not(expr):
    return "(not %s)" % expr


class SMTWriter(object):
    def __init__(self, filename, logic="QF_LRA"):
        self.filename = filename
        self.logic = logic
        self.lines = ["(set-logic %s)" % logic]

    def declare_bool(self, name):
        self.lines.append("(declare-fun %s () Bool)" % name)

    def declare_real(self, name):
        self.lines.append("(declare-fun %s () Real)" % name)

    def assert_expr(self, expr):
        self.lines.append("(assert %s)" % expr)

    def assert_implication(self, antecedent, consequent):
        self.assert_expr("(=> %s %s)" % (antecedent, consequent))

    def finish(self):
        self.lines.append("(check-sat)")
        self.lines.append("(exit)")
        ensure_dir(os.path.dirname(self.filename))
        with open(self.filename, "w") as out:
            out.write("\n".join(self.lines))
            out.write("\n")


def declare_bool_vector(writer, prefix, count):
    for i in range(count):
        writer.declare_bool("%s%d" % (prefix, i))


def declare_real_vector(writer, prefix, count):
    for i in range(count):
        writer.declare_real("%s%d" % (prefix, i))


def assert_exactly_one(writer, bool_names):
    writer.assert_expr(smt_or(bool_names))
    for left, right in itertools.combinations(bool_names, 2):
        writer.assert_expr(smt_or([smt_not(left), smt_not(right)]))


def assert_at_most_k(writer, bool_names, k, aux_prefix):
    if k < 0:
        writer.assert_expr("false")
        return
    if k >= len(bool_names):
        return
    if k == 0:
        for name in bool_names:
            writer.assert_expr(smt_not(name))
        return

    prev = []
    for i, name in enumerate(bool_names):
        current = []
        for j in range(1, k + 1):
            aux = "%s_%d_%d" % (aux_prefix, i, j)
            writer.declare_bool(aux)
            current.append(aux)

        writer.assert_implication(name, current[0])
        if prev:
            writer.assert_implication(name, smt_not(prev[k - 1]))
            for j in range(1, k + 1):
                writer.assert_implication(prev[j - 1], current[j - 1])
            for j in range(2, k + 1):
                writer.assert_implication(smt_and([name, prev[j - 2]]), current[j - 1])
        prev = current


def assert_exactly_k(writer, bool_names, k, aux_prefix):
    assert_at_most_k(writer, bool_names, k, aux_prefix + "_atmost")
    assert_at_most_k(writer, [smt_not(name) for name in bool_names],
                     len(bool_names) - k, aux_prefix + "_atleast")


def write_mixed_linear_smt2(filename, numberOfBoolVars, numberOfRealVars, constraints):
    writer = SMTWriter(filename, "QF_LRA")
    declare_bool_vector(writer, "b", numberOfBoolVars)
    declare_real_vector(writer, "y", numberOfRealVars)
    rVars = ["y%d" % i for i in range(numberOfRealVars)]

    for constraint in constraints:
        writer.assert_expr(smt_or([smt_bool_lit(lit) for lit in constraint["bool"]]))
        convex = constraint["convex"]
        if convex is not None:
            firstVar = abs(constraint["bool"][-1]) - 1
            atom = smt_linear_atom(convex["A"], rVars, convex["b"], convex["sense"])
            writer.assert_implication("b%d" % firstVar, atom)

    writer.finish()


def smt_quadratic(Q, c, vars):
    terms = []
    for i in range(len(vars)):
        for j in range(len(vars)):
            coeff = float(Q[i, j])
            if abs(coeff) >= 1e-12:
                terms.append("(* %s %s %s)" % (smt_real(coeff), vars[i], vars[j]))
    for coeff, var in zip(c, vars):
        term = smt_mul_real(coeff, var)
        if term != "0":
            terms.append(term)
    return smt_add(terms)


def write_sse_smt2(filename, numberOfBoolVars, numberOfRealVars, constraints, max_sensors_under_attack):
    writer = SMTWriter(filename, "QF_NRA")
    declare_bool_vector(writer, "b", numberOfBoolVars)
    declare_real_vector(writer, "y", numberOfRealVars)
    rVars = ["y%d" % i for i in range(numberOfRealVars)]

    for constraint in constraints:
        firstVar = constraint["bool"][0]
        convex = constraint["convex"]
        atom = "(< %s %s)" % (
            smt_quadratic(convex["Q"], convex["c"], rVars),
            smt_real(convex["b"]),
        )
        writer.assert_implication(smt_not("b%d" % firstVar), atom)

    assert_exactly_k(
        writer,
        ["b%d" % i for i in range(numberOfBoolVars)],
        max_sensors_under_attack,
        "card_attack",
    )
    writer.finish()
