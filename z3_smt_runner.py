import z3
import sys

# Check if the user provided a file path argument
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <path_to_smt2_file>")
    sys.exit(1)

file_path = sys.argv[1]

s = z3.Solver()
s.add(z3.parse_smt2_file(file_path))

res = s.check()
print("Satisfiability:", res)

if res == z3.sat:
    print("Model:", s.model())