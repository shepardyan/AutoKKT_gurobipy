from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import gurobipy as gp
from gurobipy import GRB

from kkt import add_kkt_reformulation


EPS = 1e-6


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


def build_small_rostering_case(seed: int) -> tuple[RosteringCase, list[float], int]:
    import random

    rng = random.Random(seed)
    T = 5
    I = 3
    J = 2
    N = 4

    case = RosteringCase(
        T=T,
        I=I,
        J=J,
        N=N,
        c=[[5 + rng.randint(0, 5) for _ in range(T)] for _ in range(I)],
        f=[[15 + rng.randint(0, 5) for _ in range(T)] for _ in range(J)],
        h=[[4 + rng.random() * 2 for _ in range(T)] for _ in range(J)],
        M=[35 + rng.randint(0, 8) for _ in range(T)],
        l=[1 + rng.randint(0, 1) for _ in range(I)],
        u=[3 + rng.randint(0, 1) for _ in range(I)],
        a=[1 for _ in range(J)],
        b=[2 + rng.randint(0, 1) for _ in range(J)],
    )
    dt = [20 + rng.randint(0, 8) for _ in range(T)]
    gamma = 2
    return case, dt, gamma


def solve_deterministic_master(case: RosteringCase, demand: list[float]) -> tuple[list[list[int]], list[list[int]], float]:
    m = gp.Model("rostering_mip")
    m.Params.OutputFlag = 0

    x = m.addVars(case.I, case.T, vtype=GRB.BINARY, name="x")
    y = m.addVars(case.J, case.T, vtype=GRB.BINARY, name="y")
    z = m.addVars(case.J, case.T, lb=0.0, name="z")
    w = m.addVars(case.T, lb=0.0, name="w")

    for t in range(case.T - 2):
        for i in range(case.I):
            m.addConstr(x[i, t] + x[i, t + 1] + x[i, t + 2] <= 2, name=f"full_rest[{i},{t}]")
    for t in range(case.T - 1):
        for j in range(case.J):
            m.addConstr(y[j, t] + y[j, t + 1] <= 1, name=f"part_rest[{j},{t}]")
    for i in range(case.I):
        m.addConstr(gp.quicksum(x[i, t] for t in range(case.T)) >= case.l[i], name=f"x_lb[{i}]")
        m.addConstr(gp.quicksum(x[i, t] for t in range(case.T)) <= case.u[i], name=f"x_ub[{i}]")
    for j in range(case.J):
        m.addConstr(gp.quicksum(y[j, t] for t in range(case.T)) >= case.a[j], name=f"y_lb[{j}]")
        m.addConstr(gp.quicksum(y[j, t] for t in range(case.T)) <= case.b[j], name=f"y_ub[{j}]")
    for j in range(case.J):
        for t in range(case.T):
            m.addConstr(z[j, t] <= case.N * y[j, t], name=f"link[{j},{t}]")
    for t in range(case.T):
        m.addConstr(
            case.N * gp.quicksum(x[i, t] for i in range(case.I))
            + gp.quicksum(z[j, t] for j in range(case.J))
            + w[t]
            >= demand[t],
            name=f"cover[{t}]",
        )

    obj = (
        gp.quicksum(case.c[i][t] * x[i, t] for i in range(case.I) for t in range(case.T))
        + gp.quicksum(case.f[j][t] * y[j, t] + case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T))
        + gp.quicksum(case.M[t] * w[t] for t in range(case.T))
    )
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Master MIP solve failed with status {m.Status}")

    x_val = [[int(round(x[i, t].X)) for t in range(case.T)] for i in range(case.I)]
    y_val = [[int(round(y[j, t].X)) for t in range(case.T)] for j in range(case.J)]
    return x_val, y_val, m.ObjVal


def solve_inner_lp(case: RosteringCase, x_val: list[list[int]], y_val: list[list[int]], dt: list[float], g: list[int]) -> float:
    xi = [0.05 * v for v in dt]

    m = gp.Model("inner_lp")
    m.Params.OutputFlag = 0

    z = m.addVars(case.J, case.T, lb=0.0, name="z")
    w = m.addVars(case.T, lb=0.0, name="w")

    for j in range(case.J):
        for t in range(case.T):
            m.addConstr(z[j, t] <= case.N * y_val[j][t], name=f"link[{j},{t}]")
    for t in range(case.T):
        demand = dt[t] + xi[t] * g[t]
        m.addConstr(
            case.N * sum(x_val[i][t] for i in range(case.I))
            + gp.quicksum(z[j, t] for j in range(case.J))
            + w[t]
            >= demand,
            name=f"cover[{t}]",
        )

    obj = gp.quicksum(case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T)) + gp.quicksum(
        case.M[t] * w[t] for t in range(case.T)
    )
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Inner LP solve failed with status {m.Status}")
    return m.ObjVal


def brute_force_adversary(case: RosteringCase, x_val: list[list[int]], y_val: list[list[int]], dt: list[float], gamma: int):
    best_val = -float("inf")
    best_g = None
    for g in _budgeted_binaries(case.T, gamma):
        value = solve_inner_lp(case, x_val, y_val, dt, g)
        if value > best_val + EPS:
            best_val = value
            best_g = g
    if best_g is None:
        raise RuntimeError("No feasible uncertainty vector found.")
    return best_val, best_g


def solve_adversary_with_kkt(case: RosteringCase, x_val: list[list[int]], y_val: list[list[int]], dt: list[float], gamma: int):
    xi = [0.05 * v for v in dt]

    m = gp.Model("inner_kkt")
    m.Params.OutputFlag = 0

    g = m.addVars(case.T, vtype=GRB.BINARY, name="g")
    z = m.addVars(case.J, case.T, lb=0.0, name="z")
    w = m.addVars(case.T, lb=0.0, name="w")

    m.addConstr(g.sum() <= gamma, name="unc_budget")

    constrs = []
    for j in range(case.J):
        for t in range(case.T):
            constrs.append(m.addConstr(z[j, t] <= case.N * y_val[j][t], name=f"link[{j},{t}]"))
    for t in range(case.T):
        constrs.append(
            m.addConstr(
                case.N * sum(x_val[i][t] for i in range(case.I))
                + gp.quicksum(z[j, t] for j in range(case.J))
                + w[t]
                >= dt[t] + xi[t] * g[t],
                name=f"cover[{t}]",
            )
        )

    inner_obj = gp.quicksum(case.h[j][t] * z[j, t] for j in range(case.J) for t in range(case.T)) + gp.quicksum(
        case.M[t] * w[t] for t in range(case.T)
    )
    details = add_kkt_reformulation(
        m,
        constrs,
        inner_obj,
        param_vars=[g[t] for t in range(case.T)],
        decision_vars=[z[j, t] for j in range(case.J) for t in range(case.T)] + [w[t] for t in range(case.T)],
        name="inner",
    )
    m.setObjective(inner_obj, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"KKT MILP solve failed with status {m.Status}")

    g_val = [int(round(g[t].X)) for t in range(case.T)]
    return m.ObjVal, g_val, details


def verify_hidden_equality_conversion():
    m = gp.Model("hidden_eq")
    m.Params.OutputFlag = 0

    p = m.addVar(vtype=GRB.BINARY, name="p")
    x = m.addVar(lb=-GRB.INFINITY, name="x")

    constrs = [
        m.addConstr(x <= 2.0 + p, name="ub"),
        m.addConstr(x >= 2.0 + p, name="lb"),
    ]
    details = add_kkt_reformulation(m, constrs, x, param_vars=[p], decision_vars=[x], name="eqcheck")
    m.setObjective(x, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Equality conversion test failed with status {m.Status}")
    if len(details.equality_rows) != 1:
        raise AssertionError(f"Expected 1 equality row after opposite-row detection, got {len(details.equality_rows)}")
    if details.transferred_inequalities_to_equalities != 2:
        raise AssertionError("Opposite inequalities were not transferred to an equality as in YALMIP.")
    if abs(x.X - 3.0) > EPS or int(round(p.X)) != 1:
        raise AssertionError("Hidden equality test produced an unexpected solution.")


def verify_quadratic_stationarity():
    m = gp.Model("quadratic_check")
    m.Params.OutputFlag = 0

    p = m.addVar(vtype=GRB.BINARY, name="p")
    x = m.addVar(lb=0.0, name="x")

    constrs = [m.addConstr(x >= 0.5, name="lb")]
    obj = x * x - 2.0 * p * x + p * p
    add_kkt_reformulation(m, constrs, obj, param_vars=[p], decision_vars=[x], name="quad")
    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Quadratic KKT test failed with status {m.Status}")
    if abs(m.ObjVal - 0.25) > 1e-6 or int(round(p.X)) != 0 or abs(x.X - 0.5) > 1e-6:
        raise AssertionError("Quadratic objective gradient handling is incorrect.")


def run_validation():
    verify_hidden_equality_conversion()
    verify_quadratic_stationarity()

    for seed in (7, 19, 31):
        case, dt, gamma = build_small_rostering_case(seed)
        x_val, y_val, det_obj = solve_deterministic_master(case, dt)
        brute_val, brute_g = brute_force_adversary(case, x_val, y_val, dt, gamma)
        kkt_val, kkt_g, details = solve_adversary_with_kkt(case, x_val, y_val, dt, gamma)

        if abs(brute_val - kkt_val) > 1e-5:
            raise AssertionError(
                f"Seed {seed}: KKT value mismatch. brute-force={brute_val:.8f}, kkt={kkt_val:.8f}"
            )

        print(
            f"seed={seed} deterministic_obj={det_obj:.4f} adversarial_obj={kkt_val:.4f} "
            f"g_bruteforce={brute_g} g_kkt={kkt_g} moved_eq={details.transferred_inequalities_to_equalities} "
            f"removed_dup={details.removed_duplicate_inequalities}"
        )

    print("All KKT checks passed.")


def _budgeted_binaries(length: int, gamma: int) -> Iterable[list[int]]:
    for bits in product((0, 1), repeat=length):
        if sum(bits) <= gamma:
            yield list(bits)


if __name__ == "__main__":
    run_validation()
