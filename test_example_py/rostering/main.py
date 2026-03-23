from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from test_example_py.rostering import build_random_case, rostering_mip, rostering_ro
else:
    from .rostering import build_random_case, rostering_mip, rostering_ro


def parse_args():
    parser = argparse.ArgumentParser(description="Python version of the MATLAB rostering test example.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data generation.")
    parser.add_argument("--gamma", type=int, default=3, help="Cardinality budget for the uncertainty set.")
    parser.add_argument("--verbose", action="store_true", help="Show Gurobi solver output.")
    parser.add_argument("--create-log", action="store_true", help="Write progress logs to test_example_py/rostering/log.")
    parser.add_argument("--small", action="store_true", help="Run a reduced-size instance for quick testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.small:
        case, dt = build_random_case(seed=args.seed, T=8, I=5, J=2, N=6, create_log=args.create_log)
    else:
        case, dt = build_random_case(seed=args.seed, create_log=args.create_log)

    print(f"seed={args.seed} dimensions=(T={case.T}, I={case.I}, J={case.J}, N={case.N}) gamma={args.gamma}")
    det_obj, _, _ = rostering_mip(case, dt, verbose=args.verbose)
    print(f"deterministic objective = {det_obj:.4f}")

    robust_obj, _, _ = rostering_ro(case, dt, gamma=args.gamma, verbose=args.verbose)
    print(f"robust objective = {robust_obj:.4f}")


if __name__ == "__main__":
    main()
