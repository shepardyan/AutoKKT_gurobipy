from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gurobipy as gp
from gurobipy import GRB

from kkt import add_kkt_reformulation


EPS = 1e-6


@dataclass
class ReferenceCaseResult:
    label: str
    status: int
    x: float | None
    y: float | None
    objective: float | None


@dataclass
class BilevelResult:
    u1: float
    x: float
    y: float
    bilevel_objective: float
    direct_lower_objective: float


def solve_reference_cases(*, verbose: bool = False) -> list[ReferenceCaseResult]:
    return [
        _solve_reference_case(
            "u1 = 0",
            [
                ("2*x-3*y", lambda x, y: 2 * x - 3 * y >= -12, "c1"),
                ("x+y", lambda x, y: x + y <= 14, "c2"),
                ("-3*x+y", lambda x, y: -3 * x + y <= -3, "c3"),
                ("3*x+y", lambda x, y: 3 * x + y == 30, "c4"),
            ],
            verbose=verbose,
        ),
        _solve_reference_case(
            "u1 in (0,1) U (1,+inf)",
            [
                ("2*x-3*y", lambda x, y: 2 * x - 3 * y >= -12, "c1"),
                ("x+y", lambda x, y: x + y <= 14, "c2"),
                ("-3*x+y", lambda x, y: -3 * x + y == -3, "c3"),
                ("3*x+y", lambda x, y: 3 * x + y == 30, "c4"),
            ],
            verbose=verbose,
        ),
        _solve_reference_case(
            "u1 = 1",
            [
                ("2*x-3*y", lambda x, y: 2 * x - 3 * y >= -12, "c1"),
                ("x+y", lambda x, y: x + y <= 14, "c2"),
                ("-3*x+y", lambda x, y: -3 * x + y == -3, "c3"),
                ("3*x+y", lambda x, y: 3 * x + y <= 30, "c4"),
            ],
            verbose=verbose,
        ),
    ]


def solve_lower_for_fixed_u1(u1_value: float, *, verbose: bool = False) -> tuple[float, float, float]:
    model = gp.Model("simple_bilevel_lower")
    model.Params.OutputFlag = 1 if verbose else 0

    x = model.addVar(lb=0.0, name="x")
    y = model.addVar(lb=0.0, name="y")

    model.addConstr(2 * x - 3 * y >= -12 + u1_value, name="c1")
    model.addConstr(x + y <= 14, name="c2")
    model.addConstr(-3 * x + y <= -3, name="c3")
    model.addConstr(3 * x + y <= 30, name="c4")

    objective = -x - 2 * y
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    _require_optimal(model, "simple_bilevel_lower")

    return model.ObjVal, x.X, y.X


def solve_simple_bilevel(*, verbose: bool = False) -> BilevelResult:
    model = gp.Model("simple_bilevel")
    model.Params.OutputFlag = 1 if verbose else 0

    u1 = model.addVar(lb=0.0, ub=4.0, name="u1")
    x = model.addVar(lb=0.0, name="x")
    y = model.addVar(lb=0.0, name="y")

    lower_constraints = [
        model.addConstr(2 * x - 3 * y >= -12 + u1, name="c1"),
        model.addConstr(x + y <= 14, name="c2"),
        model.addConstr(-3 * x + y <= -3, name="c3"),
        model.addConstr(3 * x + y <= 30, name="c4"),
    ]

    lower_objective = -x - 2 * y
    add_kkt_reformulation(
        model,
        lower_constraints,
        lower_objective,
        param_vars=[u1],
        decision_vars=[x, y],
        name="lower",
    )

    model.setObjective(lower_objective, GRB.MAXIMIZE)
    model.optimize()
    _require_optimal(model, "simple_bilevel")

    direct_obj, direct_x, direct_y = solve_lower_for_fixed_u1(u1.X, verbose=False)
    if abs(direct_obj - model.ObjVal) > EPS or abs(direct_x - x.X) > EPS or abs(direct_y - y.X) > EPS:
        raise AssertionError("KKT bilevel solution does not match the direct lower-level solve at the selected u1.")

    return BilevelResult(
        u1=u1.X,
        x=x.X,
        y=y.X,
        bilevel_objective=model.ObjVal,
        direct_lower_objective=direct_obj,
    )


def _solve_reference_case(
    label: str,
    constraints: list[tuple[str, Callable[[gp.Var, gp.Var], gp.TempConstr], str]],
    *,
    verbose: bool = False,
) -> ReferenceCaseResult:
    model = gp.Model(f"reference_{label}")
    model.Params.OutputFlag = 1 if verbose else 0

    x = model.addVar(lb=0.0, name="x")
    y = model.addVar(lb=0.0, name="y")

    for _, builder, name in constraints:
        model.addConstr(builder(x, y), name=name)

    objective = -x - 2 * y
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        return ReferenceCaseResult(label=label, status=model.Status, x=None, y=None, objective=None)

    return ReferenceCaseResult(label=label, status=model.Status, x=x.X, y=y.X, objective=model.ObjVal)


def _require_optimal(model: gp.Model, name: str):
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{name} solve failed with status {model.Status}")
