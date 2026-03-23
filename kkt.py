from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import gurobipy as gp
from gurobipy import GRB


TOL = 1e-9


@dataclass
class CanonicalLinearRow:
    dec_coeffs: list[float]
    param_coeffs: list[float]
    rhs: float
    source: str


@dataclass
class KKTDetails:
    primal: list[gp.Var]
    parameters: list[gp.Var]
    dual_ineq: list[gp.Var]
    dual_eq: list[gp.Var]
    slack: list[gp.Var]
    complementarity_bin: list[gp.Var]
    inequality_rows: list[CanonicalLinearRow]
    equality_rows: list[CanonicalLinearRow]
    parameter_domain: list[gp.Constr] = field(default_factory=list)
    stationarity: list[gp.Constr] = field(default_factory=list)
    removed_duplicate_inequalities: int = 0
    transferred_inequalities_to_equalities: int = 0
    info: str = ""


def add_kkt_reformulation(
    model: gp.Model,
    constraints: Sequence[gp.Constr] | gp.tupledict,
    objective,
    param_vars: Sequence[gp.Var] | gp.MVar | gp.tupledict | None = None,
    decision_vars: Sequence[gp.Var] | gp.MVar | gp.tupledict | None = None,
    *,
    include_variable_bounds: bool = True,
    name: str = "kkt",
    tol: float = TOL,
) -> KKTDetails:
    """Add a YALMIP-style KKT reformulation to ``model``.

    The inner problem is interpreted as a minimization problem in the
    variables ``decision_vars``. Integer and binary variables are allowed
    only in ``param_vars``.
    """

    model.update()

    constr_list = _flatten_constraints(constraints)
    param_list = _normalize_var_list(param_vars)
    param_keys = {_var_key(v) for v in param_list}

    if decision_vars is None:
        used = _collect_used_vars(model, constr_list, objective)
        decision_list = [v for v in used if _var_key(v) not in param_keys]
    else:
        decision_list = _normalize_var_list(decision_vars)

    decision_list = _unique_vars(decision_list)
    param_list = _unique_vars(param_list)

    if not decision_list:
        raise ValueError("No inner decision variables were identified for the KKT system.")

    decision_keys = {_var_key(v) for v in decision_list}
    overlap = decision_keys & param_keys
    if overlap:
        names = ", ".join(sorted(next(v.VarName for v in decision_list if _var_key(v) == k) for k in overlap))
        raise ValueError(f"Variables cannot be both decision and parameter variables: {names}")

    for var in decision_list:
        if var.VType != GRB.CONTINUOUS:
            raise ValueError(
                "KKT reformulation is only valid for continuous inner variables. "
                f"Move {var.VarName} outside the KKT block or fix it before derivation."
            )

    all_vars = decision_list + param_list
    all_map = {_var_key(v): idx for idx, v in enumerate(all_vars)}
    dec_count = len(decision_list)
    param_count = len(param_list)

    rows_ineq: list[tuple[list[float], float, str]] = []
    rows_eq: list[tuple[list[float], float, str]] = []

    for constr in constr_list:
        coeffs, rhs, sense = _extract_linear_constraint(model, constr, all_map)
        source = constr.ConstrName or f"constr_{constr.index}"
        if sense == GRB.LESS_EQUAL:
            rows_ineq.append((coeffs, rhs, source))
        elif sense == GRB.GREATER_EQUAL:
            rows_ineq.append(([-c for c in coeffs], -rhs, source))
        elif sense == GRB.EQUAL:
            rows_eq.append((coeffs, rhs, source))
        else:
            raise ValueError(f"Unsupported constraint sense {sense!r} in {source}.")

    if include_variable_bounds:
        for var in decision_list:
            key = _var_key(var)
            coeffs = [0.0] * len(all_vars)
            coeffs[all_map[key]] = 1.0
            if _is_finite(var.UB):
                rows_ineq.append((coeffs.copy(), float(var.UB), f"{var.VarName}_ub"))
            if _is_finite(var.LB):
                if abs(var.LB - var.UB) <= tol and _is_finite(var.UB):
                    rows_eq.append((coeffs.copy(), float(var.LB), f"{var.VarName}_fix"))
                else:
                    rows_ineq.append(([-c for c in coeffs], -float(var.LB), f"{var.VarName}_lb"))

    rows_ineq, removed_dups = _remove_duplicate_inequalities(rows_ineq, tol)
    rows_ineq, moved_eq, transferred = _move_opposite_inequalities_to_equalities(rows_ineq, tol)
    rows_eq.extend(moved_eq)

    ineq_rows = [_split_row(coeffs, rhs, source, dec_count, param_count, tol) for coeffs, rhs, source in rows_ineq]
    eq_rows = [_split_row(coeffs, rhs, source, dec_count, param_count, tol) for coeffs, rhs, source in rows_eq]

    dual_ineq: list[gp.Var] = []
    dual_eq: list[gp.Var] = []
    slacks: list[gp.Var] = []
    bins: list[gp.Var] = []
    param_domain: list[gp.Constr] = []

    active_ineq_rows: list[CanonicalLinearRow] = []
    active_eq_rows: list[CanonicalLinearRow] = []

    for i, row in enumerate(ineq_rows):
        if _all_zero(row.dec_coeffs, tol):
            param_domain.append(
                model.addConstr(_build_rhs_expression(row.rhs, row.param_coeffs, param_list) >= 0.0, name=f"{name}_dom_ineq_{i}")
            )
            continue

        lam = model.addVar(lb=0.0, name=f"{name}_lambda[{len(active_ineq_rows)}]")
        slack = model.addVar(lb=0.0, name=f"{name}_slack[{len(active_ineq_rows)}]")
        indicator = model.addVar(vtype=GRB.BINARY, name=f"{name}_delta[{len(active_ineq_rows)}]")

        slack_expr = _build_rhs_expression(row.rhs, row.param_coeffs, param_list) - gp.quicksum(
            coeff * var for coeff, var in zip(row.dec_coeffs, decision_list) if abs(coeff) > tol
        )

        model.addConstr(slack == slack_expr, name=f"{name}_slack_def[{len(active_ineq_rows)}]")
        model.addGenConstrIndicator(indicator, 0, lam == 0.0, name=f"{name}_compl_lambda[{len(active_ineq_rows)}]")
        model.addGenConstrIndicator(indicator, 1, slack == 0.0, name=f"{name}_compl_slack[{len(active_ineq_rows)}]")

        dual_ineq.append(lam)
        slacks.append(slack)
        bins.append(indicator)
        active_ineq_rows.append(row)

    for i, row in enumerate(eq_rows):
        if _all_zero(row.dec_coeffs, tol):
            param_domain.append(
                model.addConstr(_build_rhs_expression(row.rhs, row.param_coeffs, param_list) == 0.0, name=f"{name}_dom_eq_{i}")
            )
            continue

        mu = model.addVar(lb=-GRB.INFINITY, name=f"{name}_mu[{len(active_eq_rows)}]")
        model.addConstr(
            gp.quicksum(coeff * var for coeff, var in zip(row.dec_coeffs, decision_list) if abs(coeff) > tol)
            == _build_rhs_expression(row.rhs, row.param_coeffs, param_list),
            name=f"{name}_eq[{len(active_eq_rows)}]",
        )
        dual_eq.append(mu)
        active_eq_rows.append(row)

    quad_grad, param_grad, const_grad = _extract_objective_gradient(objective, decision_list, param_list, tol)
    stationarity: list[gp.Constr] = []
    for j, var in enumerate(decision_list):
        expr = gp.LinExpr()
        if abs(const_grad[j]) > tol:
            expr += const_grad[j]
        expr += gp.quicksum(coeff * pvar for coeff, pvar in zip(param_grad[j], param_list) if abs(coeff) > tol)
        expr += gp.quicksum(coeff * dvar for coeff, dvar in zip(quad_grad[j], decision_list) if abs(coeff) > tol)
        expr += gp.quicksum(row.dec_coeffs[j] * lam for row, lam in zip(active_ineq_rows, dual_ineq) if abs(row.dec_coeffs[j]) > tol)
        expr += gp.quicksum(row.dec_coeffs[j] * mu for row, mu in zip(active_eq_rows, dual_eq) if abs(row.dec_coeffs[j]) > tol)
        stationarity.append(model.addConstr(expr == 0.0, name=f"{name}_stationarity[{j}]"))

    info = "min quadratic objective" if any(any(abs(v) > tol for v in row) for row in quad_grad) else "min linear objective"
    if active_eq_rows:
        info += " s.t. Ax<=b, Ex=f"
    else:
        info += " s.t. Ax<=b"

    model.update()
    return KKTDetails(
        primal=decision_list,
        parameters=param_list,
        dual_ineq=dual_ineq,
        dual_eq=dual_eq,
        slack=slacks,
        complementarity_bin=bins,
        inequality_rows=active_ineq_rows,
        equality_rows=active_eq_rows,
        parameter_domain=param_domain,
        stationarity=stationarity,
        removed_duplicate_inequalities=removed_dups,
        transferred_inequalities_to_equalities=transferred,
        info=info,
    )


def _flatten_constraints(constraints: Sequence[gp.Constr] | gp.tupledict) -> list[gp.Constr]:
    if constraints is None:
        return []
    if isinstance(constraints, gp.tupledict):
        items = list(constraints.values())
    else:
        items = list(constraints)
    flat: list[gp.Constr] = []
    for item in items:
        if isinstance(item, gp.Constr):
            flat.append(item)
        else:
            raise TypeError(f"Unsupported constraint object {type(item)!r}.")
    return flat


def _normalize_var_list(vars_like: Sequence[gp.Var] | gp.MVar | gp.tupledict | None) -> list[gp.Var]:
    if vars_like is None:
        return []
    if isinstance(vars_like, gp.Var):
        return [vars_like]
    if isinstance(vars_like, gp.MVar):
        return [vars_like[i] for i in range(vars_like.size)]
    if isinstance(vars_like, gp.tupledict):
        return [v for _, v in vars_like.items()]
    vars_list: list[gp.Var] = []
    for item in vars_like:
        if isinstance(item, gp.Var):
            vars_list.append(item)
        elif isinstance(item, gp.MVar):
            vars_list.extend(item[i] for i in range(item.size))
        else:
            raise TypeError(f"Unsupported variable object {type(item)!r}.")
    return vars_list


def _unique_vars(vars_list: Iterable[gp.Var]) -> list[gp.Var]:
    out: list[gp.Var] = []
    seen: set[tuple[int, str]] = set()
    for var in vars_list:
        key = _var_key(var)
        if key not in seen:
            out.append(var)
            seen.add(key)
    return out


def _collect_used_vars(model: gp.Model, constraints: Sequence[gp.Constr], objective) -> list[gp.Var]:
    used: list[gp.Var] = []
    seen: set[tuple[int, str]] = set()

    def add_var(var: gp.Var) -> None:
        key = _var_key(var)
        if key not in seen:
            used.append(var)
            seen.add(key)

    for constr in constraints:
        row = model.getRow(constr)
        for i in range(row.size()):
            add_var(row.getVar(i))

    linexpr, qterms = _split_objective(objective)
    for i in range(linexpr.size()):
        add_var(linexpr.getVar(i))
    for coeff, var1, var2 in qterms:
        if abs(coeff) > TOL:
            add_var(var1)
            add_var(var2)
    return used


def _extract_linear_constraint(model: gp.Model, constr: gp.Constr, var_index: dict[tuple[int, str], int]) -> tuple[list[float], float, str]:
    row = model.getRow(constr)
    coeffs = [0.0] * len(var_index)
    for i in range(row.size()):
        var = row.getVar(i)
        key = _var_key(var)
        if key not in var_index:
            raise ValueError(f"Constraint {constr.ConstrName} contains variable {var.VarName} outside decision/parameter sets.")
        coeffs[var_index[key]] += float(row.getCoeff(i))
    return coeffs, float(constr.RHS), constr.Sense


def _remove_duplicate_inequalities(rows, tol: float):
    kept = []
    removed = 0
    best_by_lhs: dict[tuple[int, ...], tuple[list[float], float, str]] = {}
    for coeffs, rhs, source in rows:
        lhs_key = _vector_key(coeffs, tol)
        if lhs_key not in best_by_lhs:
            best_by_lhs[lhs_key] = (coeffs, rhs, source)
            continue
        _, prev_rhs, _ = best_by_lhs[lhs_key]
        if rhs < prev_rhs - tol:
            removed += 1
            best_by_lhs[lhs_key] = (coeffs, rhs, source)
        else:
            removed += 1
    kept = list(best_by_lhs.values())
    return kept, removed


def _move_opposite_inequalities_to_equalities(rows, tol: float):
    moved_eq = []
    used = [False] * len(rows)
    by_key: dict[tuple[tuple[int, ...], int], list[int]] = {}
    for idx, (coeffs, rhs, _) in enumerate(rows):
        by_key.setdefault((_vector_key(coeffs, tol), _scalar_key(rhs, tol)), []).append(idx)

    transferred = 0
    for idx, (coeffs, rhs, source) in enumerate(rows):
        if used[idx]:
            continue
        opposite = by_key.get((_vector_key([-c for c in coeffs], tol), _scalar_key(-rhs, tol)), [])
        partner = next((j for j in opposite if j != idx and not used[j]), None)
        if partner is None:
            continue
        used[idx] = True
        used[partner] = True
        moved_eq.append((coeffs, rhs, source))
        transferred += 2

    remaining = [row for flag, row in zip(used, rows) if not flag]
    return remaining, moved_eq, transferred


def _split_row(coeffs, rhs, source, dec_count, param_count, tol: float) -> CanonicalLinearRow:
    dec_coeffs = [_clean_number(v, tol) for v in coeffs[:dec_count]]
    param_coeffs = [-_clean_number(v, tol) for v in coeffs[dec_count : dec_count + param_count]]
    return CanonicalLinearRow(dec_coeffs=dec_coeffs, param_coeffs=param_coeffs, rhs=_clean_number(rhs, tol), source=source)


def _extract_objective_gradient(objective, decision_vars, param_vars, tol: float):
    linexpr, qterms = _split_objective(objective)
    dec_map = {_var_key(v): i for i, v in enumerate(decision_vars)}
    param_map = {_var_key(v): i for i, v in enumerate(param_vars)}

    n = len(decision_vars)
    m = len(param_vars)
    quad_grad = [[0.0] * n for _ in range(n)]
    param_grad = [[0.0] * m for _ in range(n)]
    const_grad = [0.0] * n

    for i in range(linexpr.size()):
        var = linexpr.getVar(i)
        coeff = float(linexpr.getCoeff(i))
        key = _var_key(var)
        if key in dec_map:
            const_grad[dec_map[key]] += coeff

    for coeff, var1, var2 in qterms:
        if abs(coeff) <= tol:
            continue
        key1 = _var_key(var1)
        key2 = _var_key(var2)
        in_dec1 = key1 in dec_map
        in_dec2 = key2 in dec_map
        in_par1 = key1 in param_map
        in_par2 = key2 in param_map

        if in_dec1 and in_dec2:
            i = dec_map[key1]
            j = dec_map[key2]
            if i == j:
                quad_grad[i][j] += 2.0 * coeff
            else:
                quad_grad[i][j] += coeff
                quad_grad[j][i] += coeff
        elif in_dec1 and in_par2:
            param_grad[dec_map[key1]][param_map[key2]] += coeff
        elif in_par1 and in_dec2:
            param_grad[dec_map[key2]][param_map[key1]] += coeff
        elif in_par1 and in_par2:
            continue
        else:
            names = f"{var1.VarName}, {var2.VarName}"
            raise ValueError(f"Objective contains variables outside decision/parameter sets: {names}")

    const_grad = [_clean_number(v, tol) for v in const_grad]
    quad_grad = [[_clean_number(v, tol) for v in row] for row in quad_grad]
    param_grad = [[_clean_number(v, tol) for v in row] for row in param_grad]
    return quad_grad, param_grad, const_grad


def _split_objective(objective):
    if isinstance(objective, gp.Var):
        expr = gp.LinExpr(objective)
        return expr, []
    if isinstance(objective, (int, float)):
        return gp.LinExpr(float(objective)), []
    if isinstance(objective, gp.LinExpr):
        return objective, []
    if isinstance(objective, gp.QuadExpr):
        linexpr = objective.getLinExpr()
        qterms = []
        for i in range(objective.size()):
            qterms.append((float(objective.getCoeff(i)), objective.getVar1(i), objective.getVar2(i)))
        return linexpr, qterms
    raise TypeError(f"Unsupported objective type {type(objective)!r}.")


def _build_rhs_expression(rhs: float, param_coeffs: list[float], param_vars: list[gp.Var]):
    expr = gp.LinExpr(float(rhs))
    if param_vars:
        expr += gp.quicksum(coeff * var for coeff, var in zip(param_coeffs, param_vars) if abs(coeff) > TOL)
    return expr


def _vector_key(values: Sequence[float], tol: float) -> tuple[int, ...]:
    scale = max(tol, 1e-12)
    return tuple(int(round(v / scale)) for v in values)


def _scalar_key(value: float, tol: float) -> int:
    scale = max(tol, 1e-12)
    return int(round(value / scale))


def _all_zero(values: Sequence[float], tol: float) -> bool:
    return all(abs(v) <= tol for v in values)


def _clean_number(value: float, tol: float) -> float:
    return 0.0 if abs(value) <= tol else float(value)


def _is_finite(value: float) -> bool:
    return value < GRB.INFINITY and value > -GRB.INFINITY


def _var_key(var: gp.Var) -> tuple[int, str]:
    return var.index, var.VarName
