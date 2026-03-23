# KKT Reformulation for Gurobi

This repository contains a small, focused Python implementation of a YALMIP-style KKT reformulation for continuous inner optimization problems built with `gurobipy`.

The main entry point is [`kkt.py`](kkt.py), which exposes `add_kkt_reformulation(...)`. The repository also includes:

- [`main.py`](main.py): validation and regression checks on small instances
- [`test_example_py/rostering/rostering.py`](test_example_py/rostering/rostering.py): a larger rostering example with robust optimization structure
- [`test_example_py/bilevel/bilevel.py`](test_example_py/bilevel/bilevel.py): a compact bilevel example plus reference LP cases
- [`KKT_INTEGRATION.md`](KKT_INTEGRATION.md): a more detailed integration guide

## What It Does

`add_kkt_reformulation(...)` converts a continuous inner minimization problem into explicit KKT conditions:

- primal feasibility
- dual feasibility
- complementarity
- stationarity

This is useful for models of the form:

```text
max_y min_x f(x, y)
s.t. g(x, y) <= 0
     h(x, y) = 0
```

where:

- `x` is continuous
- `y` contains outer decisions or parameter variables
- the inner constraints are linear
- the inner objective is linear or quadratic in `x`

The implementation is intentionally aligned with the preprocessing pattern used by YALMIP's `kkt.m`, including:

- normalization of `>=` rows into `<=`
- inclusion of explicit variable bounds
- duplicate inequality cleanup
- conversion of opposite inequalities into equalities
- separation of pure parameter-domain constraints

## Scope and Limitations

This helper is valid when the inner problem is:

- a minimization problem
- linear or quadratic in the inner variables
- linear in the constraints
- continuous in the inner decision variables

It is not valid for:

- inner MILPs with binary or integer decision variables
- nonlinear inner constraints
- objectives beyond quadratic in the inner variables

If an inner decision variable is not continuous, the helper raises an error instead of silently building an invalid reformulation.

## Repository Layout

```text
.
├── kkt.py
├── main.py
├── KKT_INTEGRATION.md
└── test_example_py
    ├── bilevel
    │   ├── bilevel.py
    │   └── main.py
    └── rostering
        ├── main.py
        └── rostering.py
```

## Requirements

- Python 3.11+ recommended
- Gurobi installed locally
- `gurobipy` available in the active Python environment
- a valid Gurobi license

## Quick Start

Run the small validation script:

```bash
python main.py
```

This script:

- checks hidden-equality detection
- checks quadratic stationarity handling
- compares the KKT adversary result against brute-force enumeration on small rostering instances

Run the translated rostering example:

```bash
python -m test_example_py.rostering.main --small --seed 0 --gamma 3
```

Optional flags:

- `--verbose`: show Gurobi solver output
- `--create-log`: write progress logs
- `--small`: run a smaller instance for quick testing

Run the simple bilevel example:

```bash
python -m test_example_py.bilevel.main
```

## Minimal Usage

```python
import gurobipy as gp
from gurobipy import GRB

from kkt import add_kkt_reformulation

m = gp.Model()
y = m.addVar(vtype=GRB.BINARY, name="y")
x = m.addVars(2, lb=0.0, name="x")

inner_constraints = [
    m.addConstr(x[0] + x[1] >= 1 + y, name="demand"),
    m.addConstr(x[0] <= 2, name="ub0"),
]

inner_obj = 3 * x[0] + 2 * x[1]

details = add_kkt_reformulation(
    m,
    inner_constraints,
    inner_obj,
    param_vars=[y],
    decision_vars=[x[0], x[1]],
    name="inner",
)

m.setObjective(inner_obj, GRB.MAXIMIZE)
m.optimize()
```

The returned `details` object exposes generated dual variables, slacks, complementarity binaries, processed rows, and stationarity constraints for inspection and debugging.

## Additional Documentation

For the full modeling pattern and integration notes, read [`KKT_INTEGRATION.md`](KKT_INTEGRATION.md).
