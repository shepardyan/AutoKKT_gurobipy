# KKT Integration Guide

This document explains how to integrate the KKT reformulation in this repository into optimization models built with `gurobipy`.

The implementation lives in [kkt.py](./kkt.py). A complete working example based on the MATLAB rostering model lives in [test_example_py/rostering/rostering.py](./test_example_py/rostering/rostering.py).

## 1. What This KKT Reformulation Does

`add_kkt_reformulation(...)` converts a continuous inner minimization problem into a set of primal feasibility, dual feasibility, complementarity, and stationarity constraints.

This is useful for problems of the form

```text
max_y min_x f(x, y)
s.t. g(x, y) <= 0
     h(x, y) = 0
```

where:

- `x` are the inner decision variables
- `y` are outer variables or parameters
- the inner problem is an LP or QP in `x`

After reformulation, the inner optimization is replaced by constraints, so the full model can be solved directly as a MIP or MIQP.

## 2. Validity Conditions

The implementation is intentionally aligned with YALMIP's `kkt.m`.

It is valid when:

- the inner problem is a minimization problem
- the inner decision variables are continuous
- the inner constraints are linear
- the inner objective is linear or quadratic in the inner decision variables
- binary or integer variables appear only as outer variables / parameters, not as inner KKT decision variables

It is not valid when:

- the inner decision variables are binary or integer
- the inner constraints are nonlinear
- the inner objective is nonlinear beyond quadratic

If an inner decision variable is integer, `add_kkt_reformulation(...)` raises an error instead of building an invalid KKT system.

## 3. API

```python
from kkt import add_kkt_reformulation

details = add_kkt_reformulation(
    model,
    constraints,
    objective,
    param_vars=None,
    decision_vars=None,
    include_variable_bounds=True,
    name="kkt",
)
```

Arguments:

- `model`: a `gurobipy.Model`
- `constraints`: a sequence of linear `gurobipy.Constr` objects defining the inner problem
- `objective`: the inner minimization objective, as a `LinExpr`, `QuadExpr`, or variable
- `param_vars`: variables treated as parameters in the inner problem
- `decision_vars`: continuous inner variables
- `include_variable_bounds`: whether variable bounds should be added as KKT rows
- `name`: prefix for generated variables and constraints

Return value:

- a `KKTDetails` object containing the generated dual variables, slacks, complementarity binaries, processed rows, and stationarity constraints

## 4. Core Modeling Pattern

The standard integration pattern is:

1. Build the outer model.
2. Add the outer variables first.
3. Add the inner variables next.
4. Add the inner constraints as normal gurobipy constraints and keep references to them.
5. Build the inner objective.
6. Call `add_kkt_reformulation(...)`.
7. Set the outer objective and solve the full model.

### Minimal Example

```python
import gurobipy as gp
from gurobipy import GRB

from kkt import add_kkt_reformulation

m = gp.Model()
m.Params.OutputFlag = 0

# Outer / parameter variable
y = m.addVar(vtype=GRB.BINARY, name="y")

# Inner continuous decision variables
x = m.addVars(2, lb=0.0, name="x")

# Inner constraints
inner_constrs = []
inner_constrs.append(m.addConstr(x[0] + x[1] >= 1 + y, name="demand"))
inner_constrs.append(m.addConstr(x[0] <= 2, name="ub0"))

# Inner minimization objective
inner_obj = 3 * x[0] + 2 * x[1]

# Replace min_x { inner_obj } by KKT conditions
details = add_kkt_reformulation(
    m,
    inner_constrs,
    inner_obj,
    param_vars=[y],
    decision_vars=[x[0], x[1]],
    name="inner",
)

# Outer objective
m.setObjective(inner_obj, GRB.MAXIMIZE)
m.optimize()
```

## 5. How Parameters Are Handled

Parameter variables are variables that appear in the inner model but are not optimized by the KKT system.

Typical parameter variables are:

- adversarial uncertainty binaries
- upper-level decision variables in bilevel models
- fixed binary pattern decisions passed into a continuous recourse problem

In YALMIP terms, these correspond to the `parametricVariables` argument.

In this implementation, any coefficient involving parameter variables is moved to the right-hand side of the processed KKT rows. The stationarity equations also keep the parameter-dependent part of the objective gradient.

## 6. What the Implementation Generates

For inequality rows:

- one nonnegative dual variable `lambda`
- one nonnegative slack variable
- one binary indicator for complementarity

Complementarity is enforced by indicator constraints:

- if the indicator selects the dual-active case, the slack is forced to zero
- otherwise the dual is forced to zero

For equality rows:

- one free dual variable `mu`

For all inner decision variables:

- one stationarity equation

## 7. YALMIP-Aligned Preprocessing

Before building the KKT system, the implementation mirrors key steps from the MATLAB version:

- normalizes `>=` rows into `<=`
- includes explicit variable bounds as constraints
- removes duplicated or weaker duplicate inequalities
- detects opposite inequalities and converts them into equalities
- separates rows with no inner decision-variable coefficients into pure parameter-domain constraints

This matters because the MATLAB version relies on the same cleanup before forming dual variables and stationarity conditions.

## 8. Recommended Integration Strategy

Use the KKT reformulation only for the continuous inner problem.

Good pattern:

- outer model may contain binary or integer variables
- inner recourse model is continuous LP/QP
- call `add_kkt_reformulation(...)` on that inner block

Bad pattern:

- inner block still contains binary or integer decision variables

If the true inner problem is MILP, you must first reformulate it by fixing or enumerating the inner integer structure, or use another exact reformulation. KKT conditions alone are not valid for a nonconvex discrete inner problem.

## 9. Example from This Repository

See [test_example_py/rostering/rostering.py](./test_example_py/rostering/rostering.py).

In the translated `InCCG` step:

- `g` is the outer uncertainty binary vector
- `z` and `w` are the continuous inner recourse variables
- a fixed binary schedule `y_cut` is treated as data
- the inner LP objective is minimized in `z, w`
- the KKT system is added with `g` as `param_vars`

The integration call is:

```python
add_kkt_reformulation(
    model,
    subproblem_constraints,
    inner_obj,
    param_vars=[g[t] for t in range(case.T)],
    decision_vars=[z[j, t] for j in range(case.J) for t in range(case.T)] + [w[t] for t in range(case.T)],
    name=f"inner_{k}",
)
```

## 10. Common Mistakes

- Passing integer inner variables in `decision_vars`
- Forgetting to keep references to the inner constraints
- Using the wrong objective sense: the KKT builder assumes an inner minimization problem
- Setting the full model objective before the KKT block is built and then accidentally reusing the wrong expression
- Expecting KKT to be valid for a discrete inner problem

## 11. Debugging Tips

If the integrated model behaves unexpectedly:

- verify the inner problem alone first
- confirm every variable in `decision_vars` is continuous
- confirm every outer variable that appears in the inner model is listed in `param_vars`
- inspect `details.inequality_rows` and `details.equality_rows`
- inspect `details.stationarity`
- compare the KKT reformulation result against direct solve or brute-force enumeration on a small instance

The validation script in [main.py](./main.py) does exactly this on reduced examples.

## 12. Verification Workflow

Recommended verification steps for a new model:

1. Solve the inner problem directly for a fixed parameter vector.
2. Build a model with the KKT reformulation for the same fixed parameter vector.
3. Check that primal variables and objective values match.
4. Only then embed the KKT block into the larger outer model.

For binary uncertainty sets or small outer domains, brute-force enumeration is the cleanest correctness check.

## 13. Practical Notes

- Complementarity is modeled with indicators, so the reformulated model is usually a MIP even if the original inner problem is continuous.
- If the inner objective is quadratic, the final outer model may become an MIQP.
- Tight explicit bounds on inner variables generally help numerics.
- The helper includes finite variable bounds automatically unless `include_variable_bounds=False`.

## 14. Files to Read

- Main KKT implementation: [kkt.py](./kkt.py)
- Validation and regression checks: [main.py](./main.py)
- Python translation of the MATLAB rostering example: [test_example_py/rostering/rostering.py](./test_example_py/rostering/rostering.py)
- Rostering example runner: [test_example_py/rostering/main.py](./test_example_py/rostering/main.py)
- Simple bilevel example: [test_example_py/bilevel/bilevel.py](./test_example_py/bilevel/bilevel.py)
- Bilevel example runner: [test_example_py/bilevel/main.py](./test_example_py/bilevel/main.py)
