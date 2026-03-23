from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

from kkt import add_kkt_reformulation

TIGHT_GUROBI_PARAMS = {
    "MIPGap": 1e-9,
    "MIPGapAbs": 1e-9,
    "FeasibilityTol": 1e-9,
    "IntFeasTol": 1e-9,
    "OptimalityTol": 1e-9,
    "NumericFocus": 2,
    "Method": 0,
    "Presolve": 2,
    "Aggregate": 0,
    "Seed": 1,
}


@dataclass
class RosteringCase:
    T: int
    I: int
    J: int
    N: int
    c: list[list[float]]
    f: list[list[float]]
    h: list[list[float]]
    M: list[float]
    l: list[int]
    u: list[int]
    a: list[int]
    b: list[int]
    log_path: str | None = None


def build_random_case(
    *,
    seed: int = 0,
    T: int = 21,
    I: int = 12,
    J: int = 3,
    N: int = 8,
    create_log: bool = False,
    log_dir: str | Path = "test_example_py/rostering/log",
) -> tuple[RosteringCase, list[float]]:
    import random
    from datetime import datetime

    rng = random.Random(seed)
    log_path = None
    if create_log:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_path = str(log_dir / f"{timestamp}.txt")
        Path(log_path).write_text("")

    full_time_max = (2 * T + 2) // 3
    part_time_max = (T + 1) // 2
    l_low = min(4, full_time_max)
    l_high = min(8, full_time_max)
    a_low = min(2, part_time_max)
    a_high = min(4, part_time_max)

    l_values = [rng.randint(max(1, l_low), max(1, l_high)) for _ in range(I)]
    u_values = [rng.randint(l_values[i], max(l_values[i], min(14, full_time_max))) for i in range(I)]
    a_values = [rng.randint(max(1, a_low), max(1, a_high)) for _ in range(J)]
    b_values = [rng.randint(a_values[j], max(a_values[j], min(6, part_time_max))) for j in range(J)]

    case = RosteringCase(
        T=T,
        I=I,
        J=J,
        N=N,
        c=[[10.0 * rng.random() + 5.0 for _ in range(T)] for _ in range(I)],
        f=[[10.0 * rng.random() + 20.0 for _ in range(T)] for _ in range(J)],
        h=[[4.0 * rng.random() + 4.0 for _ in range(T)] for _ in range(J)],
        M=[10.0 * rng.random() + 40.0 for _ in range(T)],
        l=l_values,
        u=u_values,
        a=a_values,
        b=b_values,
        log_path=log_path,
    )
    dt = [50.0 + 30.0 * rng.random() for _ in range(T)]
    return case, dt


def rostering_mip(case: RosteringCase, d: list[float], *, verbose: bool = False, gurobi_params: dict | None = None):
    model = gp.Model("rostering_mip")
    model.Params.OutputFlag = 1 if verbose else 0
    _apply_gurobi_params(model, gurobi_params)

    x = model.addVars(case.I, case.T, vtype=GRB.BINARY, name="x")
    y = model.addVars(case.J, case.T, vtype=GRB.BINARY, name="y")
    z = model.addVars(case.J, case.T, lb=0.0, name="z")
    w = model.addVars(case.T, lb=0.0, name="w")

    _add_regular_staff_constraints(model, case, x)
    _add_part_time_pattern_constraints(model, case, y)
    _add_shift_bounds(model, case, x, y)
    _add_link_constraints(model, case, y, z)
    _add_coverage_constraints(model, case, x, z, w, d)

    objective = (
        gp.quicksum(case.c[i][t] * x[i, t] for i in range(case.I) for t in range(case.T))
        + gp.quicksum(case.f[j][t] * y[j, t] + case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T))
        + gp.quicksum(case.M[t] * w[t] for t in range(case.T))
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    _require_optimal(model, "rostering_mip")

    x_value = _extract_binary_matrix(x, case.I, case.T)
    y_value = _extract_binary_matrix(y, case.J, case.T)
    return model.ObjVal, x_value, y_value


def mp(case: RosteringCase, cuts: list[list[float]], *, verbose: bool = False, gurobi_params: dict | None = None):
    na = len(cuts)
    model = gp.Model("MP")
    model.Params.OutputFlag = 1 if verbose else 0
    _apply_gurobi_params(model, gurobi_params)

    x = model.addVars(case.I, case.T, vtype=GRB.BINARY, name="x")
    y = model.addVars(case.J, case.T, na, vtype=GRB.BINARY, name="y")
    z = model.addVars(case.J, case.T, na, lb=0.0, name="z")
    w = model.addVars(case.T, na, lb=0.0, name="w")
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")

    _add_regular_staff_constraints(model, case, x)
    for i in range(case.I):
        model.addConstr(gp.quicksum(x[i, t] for t in range(case.T)) >= case.l[i], name=f"x_lb[{i}]")
        model.addConstr(gp.quicksum(x[i, t] for t in range(case.T)) <= case.u[i], name=f"x_ub[{i}]")

    for cut_idx, demand in enumerate(cuts):
        model.addConstr(
            eta
            >= gp.quicksum(
                case.f[j][t] * y[j, t, cut_idx] + case.h[j][t] * z[j, t, cut_idx]
                for j in range(case.J)
                for t in range(case.T)
            )
            + gp.quicksum(case.M[t] * w[t, cut_idx] for t in range(case.T)),
            name=f"eta_cut[{cut_idx}]",
        )

        for t in range(case.T - 1):
            for j in range(case.J):
                model.addConstr(y[j, t, cut_idx] + y[j, t + 1, cut_idx] <= 1, name=f"part_rest[{j},{t},{cut_idx}]")
        for j in range(case.J):
            model.addConstr(
                gp.quicksum(y[j, t, cut_idx] for t in range(case.T)) >= case.a[j], name=f"y_lb[{j},{cut_idx}]"
            )
            model.addConstr(
                gp.quicksum(y[j, t, cut_idx] for t in range(case.T)) <= case.b[j], name=f"y_ub[{j},{cut_idx}]"
            )
            for t in range(case.T):
                model.addConstr(z[j, t, cut_idx] <= case.N * y[j, t, cut_idx], name=f"link[{j},{t},{cut_idx}]")
        for t in range(case.T):
            model.addConstr(
                case.N * gp.quicksum(x[i, t] for i in range(case.I))
                + gp.quicksum(z[j, t, cut_idx] for j in range(case.J))
                + w[t, cut_idx]
                >= demand[t],
                name=f"cover[{t},{cut_idx}]",
            )

    objective = gp.quicksum(case.c[i][t] * x[i, t] for i in range(case.I) for t in range(case.T)) + eta
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    _require_optimal(model, "MP")

    return model.ObjVal, _extract_binary_matrix(x, case.I, case.T)


def in_sp(case: RosteringCase, x_value: list[list[int]], d: list[float], *, verbose: bool = False, gurobi_params: dict | None = None):
    model = gp.Model("InSP")
    model.Params.OutputFlag = 1 if verbose else 0
    _apply_gurobi_params(model, gurobi_params)

    y = model.addVars(case.J, case.T, vtype=GRB.BINARY, name="y")
    z = model.addVars(case.J, case.T, lb=0.0, name="z")
    w = model.addVars(case.T, lb=0.0, name="w")

    _add_part_time_pattern_constraints(model, case, y)
    for j in range(case.J):
        model.addConstr(gp.quicksum(y[j, t] for t in range(case.T)) >= case.a[j], name=f"y_lb[{j}]")
        model.addConstr(gp.quicksum(y[j, t] for t in range(case.T)) <= case.b[j], name=f"y_ub[{j}]")
        for t in range(case.T):
            model.addConstr(z[j, t] <= case.N * y[j, t], name=f"link[{j},{t}]")
    _add_coverage_constraints(model, case, x_value, z, w, d)

    objective = gp.quicksum(case.f[j][t] * y[j, t] + case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T)) + gp.quicksum(
        case.M[t] * w[t] for t in range(case.T)
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    _require_optimal(model, "InSP")

    return model.ObjVal, _extract_binary_matrix(y, case.J, case.T)


def in_ccg(
    case: RosteringCase,
    x_value: list[list[int]],
    dt: list[float],
    gamma: int | None = None,
    *,
    t1: int | None = None,
    rho1: int | None = None,
    rho2: int | None = None,
    delta: float = 1e-3,
    verbose: bool = False,
    gurobi_params: dict | None = None,
):
    if gamma is None and None in (t1, rho1, rho2):
        raise ValueError("Provide either gamma or (t1, rho1, rho2).")

    xi = [0.05 * value for value in dt]
    in_lb = -float("inf")
    in_ub = float("inf")
    y_cuts: list[list[list[int]]] = [[[0 for _ in range(case.T)] for _ in range(case.J)]]

    model = gp.Model("InCCG")
    model.Params.OutputFlag = 1 if verbose else 0
    _apply_gurobi_params(model, gurobi_params)

    g = model.addVars(case.T, vtype=GRB.BINARY, name="g")
    theta = model.addVar(lb=0.0, name="theta")

    if gamma is not None:
        model.addConstr(g.sum() <= gamma, name="uncertainty_budget")
    else:
        assert t1 is not None and rho1 is not None and rho2 is not None
        model.addConstr(gp.quicksum(g[t] for t in range(t1 + 2)) <= rho1, name="uncertainty_budget_1")
        model.addConstr(gp.quicksum(g[t] for t in range(t1 - 1, case.T)) <= rho2, name="uncertainty_budget_2")

    k = 0
    while in_ub - in_lb > delta:
        y_cut = y_cuts[k]
        z = model.addVars(case.J, case.T, lb=0.0, name=f"z_{k}")
        w = model.addVars(case.T, lb=0.0, name=f"w_{k}")

        model.addConstr(
            theta
            <= gp.quicksum(case.f[j][t] * y_cut[j][t] + case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T))
            + gp.quicksum(case.M[t] * w[t] for t in range(case.T)),
            name=f"theta_cut[{k}]",
        )

        subproblem_constraints = []
        for j in range(case.J):
            for t in range(case.T):
                subproblem_constraints.append(
                    model.addConstr(z[j, t] <= case.N * y_cut[j][t], name=f"link[{j},{t},{k}]")
                )
        for t in range(case.T):
            subproblem_constraints.append(
                model.addConstr(
                    case.N * sum(x_value[i][t] for i in range(case.I))
                    + gp.quicksum(z[j, t] for j in range(case.J))
                    + w[t]
                    >= dt[t] + xi[t] * g[t],
                    name=f"cover[{t},{k}]",
                )
            )

        inner_obj = gp.quicksum(case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T)) + gp.quicksum(
            case.M[t] * w[t] for t in range(case.T)
        )
        add_kkt_reformulation(
            model,
            subproblem_constraints,
            inner_obj,
            param_vars=[g[t] for t in range(case.T)],
            decision_vars=[z[j, t] for j in range(case.J) for t in range(case.T)] + [w[t] for t in range(case.T)],
            name=f"inner_{k}",
        )

        model.setObjective(theta, GRB.MAXIMIZE)
        model.optimize()
        _require_optimal(model, "InCCG")

        in_ub = theta.X
        d_value = [dt[t] + xi[t] * round(g[t].X) for t in range(case.T)]

        obj_sp, y_sp = in_sp(case, x_value, d_value, verbose=verbose, gurobi_params=gurobi_params)
        in_lb = max(in_lb, obj_sp)
        y_cuts.append(y_sp)

        _log(
            case,
            f"  Inner Iteration {k + 1:2d}, bound is {in_ub - in_lb:10.2f}. UB is {in_ub:10.2f}, LB is {in_lb:10.2f}",
        )
        k += 1

    return in_lb, d_value


def rostering_ro(
    case: RosteringCase,
    dt: list[float],
    gamma: int | None = None,
    *,
    t1: int | None = None,
    rho1: int | None = None,
    rho2: int | None = None,
    delta: float = 1e-3,
    verbose: bool = False,
    gurobi_params: dict | None = None,
):
    _log(case, "Nested C&CG starts")

    lb = -float("inf")
    ub = float("inf")
    cuts = [list(dt)]
    k = 0

    while ub - lb > delta:
        k += 1
        obj_outer, x_value = mp(case, cuts, verbose=verbose, gurobi_params=gurobi_params)
        lb = obj_outer

        if gamma is not None:
            obj_inner, d_value = in_ccg(
                case, x_value, dt, gamma=gamma, delta=delta, verbose=verbose, gurobi_params=gurobi_params
            )
        else:
            obj_inner, d_value = in_ccg(
                case,
                x_value,
                dt,
                t1=t1,
                rho1=rho1,
                rho2=rho2,
                delta=delta,
                verbose=verbose,
                gurobi_params=gurobi_params,
            )

        cuts.append(d_value)
        ub = min(ub, obj_inner + _first_stage_cost(case, x_value))
        _log(case, f"Outer Iteration {k:2d}, bound is {ub - lb:10.2f}. UB is {ub:10.2f}, LB is {lb:10.2f}")

    final_cut = cuts[max(k - 1, 0)]
    _, y_value = in_sp(case, x_value, final_cut, verbose=verbose, gurobi_params=gurobi_params)
    return ub, x_value, y_value


def MP(case: RosteringCase, cuts: list[list[float]], *, verbose: bool = False, gurobi_params: dict | None = None):
    return mp(case, cuts, verbose=verbose, gurobi_params=gurobi_params)


def InSP(case: RosteringCase, x_value: list[list[int]], d: list[float], *, verbose: bool = False, gurobi_params: dict | None = None):
    return in_sp(case, x_value, d, verbose=verbose, gurobi_params=gurobi_params)


def InCCG(
    case: RosteringCase,
    x_value: list[list[int]],
    dt: list[float],
    gamma: int | None = None,
    *,
    t1: int | None = None,
    rho1: int | None = None,
    rho2: int | None = None,
    delta: float = 1e-3,
    verbose: bool = False,
    gurobi_params: dict | None = None,
):
    return in_ccg(
        case,
        x_value,
        dt,
        gamma=gamma,
        t1=t1,
        rho1=rho1,
        rho2=rho2,
        delta=delta,
        verbose=verbose,
        gurobi_params=gurobi_params,
    )


def _add_regular_staff_constraints(model: gp.Model, case: RosteringCase, x):
    for t in range(case.T - 2):
        for i in range(case.I):
            model.addConstr(x[i, t] + x[i, t + 1] + x[i, t + 2] <= 2, name=f"full_rest[{i},{t}]")


def _add_part_time_pattern_constraints(model: gp.Model, case: RosteringCase, y):
    for t in range(case.T - 1):
        for j in range(case.J):
            model.addConstr(y[j, t] + y[j, t + 1] <= 1, name=f"part_rest[{j},{t}]")


def _add_shift_bounds(model: gp.Model, case: RosteringCase, x, y):
    for i in range(case.I):
        model.addConstr(gp.quicksum(x[i, t] for t in range(case.T)) >= case.l[i], name=f"x_lb[{i}]")
        model.addConstr(gp.quicksum(x[i, t] for t in range(case.T)) <= case.u[i], name=f"x_ub[{i}]")
    for j in range(case.J):
        model.addConstr(gp.quicksum(y[j, t] for t in range(case.T)) >= case.a[j], name=f"y_lb[{j}]")
        model.addConstr(gp.quicksum(y[j, t] for t in range(case.T)) <= case.b[j], name=f"y_ub[{j}]")


def _add_link_constraints(model: gp.Model, case: RosteringCase, y, z):
    for j in range(case.J):
        for t in range(case.T):
            model.addConstr(z[j, t] <= case.N * y[j, t], name=f"link[{j},{t}]")


def _add_coverage_constraints(model: gp.Model, case: RosteringCase, x_or_values, z, w, d: list[float]):
    for t in range(case.T):
        x_term = _sum_x_at_t(case, x_or_values, t)
        model.addConstr(
            case.N * x_term + gp.quicksum(z[j, t] for j in range(case.J)) + w[t] >= d[t],
            name=f"cover[{t}]",
        )


def _sum_x_at_t(case: RosteringCase, x_or_values, t: int):
    if isinstance(x_or_values, gp.tupledict):
        return gp.quicksum(x_or_values[i, t] for i in range(case.I))
    return sum(x_or_values[i][t] for i in range(case.I))


def _extract_binary_matrix(var_dict, rows: int, cols: int):
    return [[int(round(var_dict[i, t].X)) for t in range(cols)] for i in range(rows)]


def _first_stage_cost(case: RosteringCase, x_value: list[list[int]]):
    return sum(case.c[i][t] * x_value[i][t] for i in range(case.I) for t in range(case.T))


def _require_optimal(model: gp.Model, name: str):
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{name} solve failed with status {model.Status}")


def _apply_gurobi_params(model: gp.Model, params: dict | None):
    if not params:
        return
    for key, value in params.items():
        setattr(model.Params, key, value)


def _log(case: RosteringCase, message: str):
    print(message)
    if case.log_path:
        with open(case.log_path, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
