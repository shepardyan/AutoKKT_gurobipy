from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from test_example_py.bilevel import solve_lower_for_fixed_u1, solve_reference_cases, solve_simple_bilevel
else:
    from . import solve_lower_for_fixed_u1, solve_reference_cases, solve_simple_bilevel


def parse_args():
    parser = argparse.ArgumentParser(description="Small bilevel example using the KKT reformulation helper.")
    parser.add_argument("--verbose", action="store_true", help="Show Gurobi solver output.")
    parser.add_argument("--u1", type=float, default=None, help="Also solve the lower-level LP directly at a fixed u1.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Reference LP cases from the MATLAB active-set check:")
    for result in solve_reference_cases(verbose=args.verbose):
        if result.objective is None:
            print(f"  {result.label}: no optimal solution (Gurobi status {result.status})")
        else:
            print(f"  {result.label}: x={result.x:.4f}, y={result.y:.4f}, objective={result.objective:.4f}")

    bilevel = solve_simple_bilevel(verbose=args.verbose)
    print("\nSimple bilevel max-min example:")
    print(f"  upper u1 = {bilevel.u1:.4f}")
    print(f"  lower x  = {bilevel.x:.4f}")
    print(f"  lower y  = {bilevel.y:.4f}")
    print(f"  bilevel objective = {bilevel.bilevel_objective:.4f}")

    if args.u1 is not None:
        lower_obj, lower_x, lower_y = solve_lower_for_fixed_u1(args.u1, verbose=args.verbose)
        print("\nDirect lower-level solve at fixed u1:")
        print(f"  u1 = {args.u1:.4f}")
        print(f"  x  = {lower_x:.4f}")
        print(f"  y  = {lower_y:.4f}")
        print(f"  objective = {lower_obj:.4f}")


if __name__ == "__main__":
    main()
