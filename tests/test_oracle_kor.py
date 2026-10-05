"""Unabhängige Orakel (anderer Rechenweg als der Demo-Code): Wartezeiten, reine Gleichgewichte und Gehorsams-Ungleichungen in reinen Python-Schleifen direkt aus der Definition, das LP für
korrelierte/grobe korrelierte Gleichgewichte dicht aufgebaut und mit `highs-ipm` gelöst, Regret-Matching-Wahrscheinlichkeiten und CE-Lücken per Schleife, Mini-Spiel per Handformel
und Gittersuche."""

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import linprog

import kor_bimatrix as bm
import kor_ce as ce
import kor_constants as C
import kor_gates as gates
import kor_learning as learn


def _wait(inst, s, i, h=None):
    h = s[i] if h is None else h
    return inst.a[h] + inst.b[h] * (inst.w[i] + sum(inst.w[j] for j in range(inst.n) if j != i and s[j] == h))


def _inst(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(2, 5)), int(rng.integers(2, 4))
    return gates.Instance(rng.uniform(1, 10, m), rng.uniform(0.3, 3, m), rng.choice([1.0, 2.0, 3.0], n), seed)


@pytest.mark.parametrize("seed", range(15))
def test_states_enumeration_and_lp_values_equal_definition_based_references(seed):
    inst = _inst(seed)
    S = list(itertools.product(range(inst.m), repeat=inst.n))
    st = ce.States(inst)
    key = {tuple(int(x) for x in st.A[k]): k for k in range(st.N)}
    for s in S:
        for i in range(inst.n):
            assert st.own[key[s], i] == pytest.approx(_wait(inst, s, i))
            assert [st.dev[key[s], i, h] for h in range(inst.m)] == pytest.approx([_wait(inst, s, i, h) for h in range(inst.m)])
    ne = [s for s in S if all(_wait(inst, s, i) <= min(_wait(inst, s, i, h) for h in range(inst.m)) + 1e-9 for i in range(inst.n))]
    en = gates.analyse(inst)
    soc = {s: sum(_wait(inst, s, i) for i in range(inst.n)) for s in S}
    assert en["n_ne"] == len(ne) and en["opt_cost"] == pytest.approx(min(soc.values()))
    assert en["ne_cost_min"] == pytest.approx(min(soc[s] for s in ne)) and en["ne_cost_max"] == pytest.approx(max(soc[s] for s in ne))
    for kind in ("ce", "cce"):
        rows = []
        for i in range(inst.n):
            for g in range(inst.m):
                for h in range(inst.m):
                    if kind == "ce" and g != h:
                        rows.append([(_wait(inst, s, i) - _wait(inst, s, i, h)) if s[i] == g else 0.0 for s in S])
            if kind == "cce":
                for h in range(inst.m):
                    rows.append([_wait(inst, s, i) - _wait(inst, s, i, h) for s in S])
        for sense, sign in (("min", 1.0), ("max", -1.0)):
            ref = linprog(sign * np.array([soc[s] for s in S]), A_ub=np.array(rows), b_ub=np.zeros(len(rows)), A_eq=np.ones((1, len(S))), b_eq=[1.0], bounds=(0, None), method="highs-ipm")
            val, sigma = ce.solve(st, kind, sense)
            assert val == pytest.approx(sign * ref.fun, rel=1e-6)
            sig = np.array([sigma[key[s]] for s in S])
            assert (np.array(rows) @ sig).max() < 1e-7 and sig.sum() == pytest.approx(1.0)
    assert en["opt_cost"] <= ce.solve(st, "cce", "min")[0] + 1e-7 <= ce.solve(st, "ce", "min")[0] + 2e-7 <= en["ne_cost_min"] + 3e-7


@pytest.mark.parametrize("seed", range(6))
def test_ce_gap_equals_loop(seed):
    inst = _inst(seed + 40)
    S = list(itertools.product(range(inst.m), repeat=inst.n))
    st = ce.States(inst)
    key = {tuple(int(x) for x in st.A[k]): k for k in range(st.N)}
    sg = np.random.default_rng(seed).dirichlet(np.ones(len(S)) * 0.3)
    sigma = np.zeros(st.N)
    for k, s in enumerate(S):
        sigma[key[s]] = sg[k]
    r_ce = r_cce = 0.0
    for i in range(inst.n):
        for h in range(inst.m):
            r_cce = max(r_cce, sum(sg[k] * (_wait(inst, s, i) - _wait(inst, s, i, h)) for k, s in enumerate(S)))
            for g in range(inst.m):
                if g != h:
                    r_ce = max(r_ce, sum(sg[k] * (_wait(inst, s, i) - _wait(inst, s, i, h)) for k, s in enumerate(S) if s[i] == g))
    assert ce.ce_gap(st, sigma) == pytest.approx((r_ce, r_cce))


@pytest.mark.parametrize("seed", range(4))
def test_regret_matching_probabilities_and_gap_curves_equal_loops(seed):
    inst = _inst(seed + 80)
    n, m, T = inst.n, inst.m, 40
    lr = learn.run_learning(inst, "rm", T, seed)
    mu = (m - 1) * max(inst.a[g] + inst.b[g] * inst.w.sum() for g in range(m))
    for t in range(1, T):
        for i in range(n):
            g = lr.assign[t - 1, i]
            q = np.array([0.0 if h == g else max(sum(lr.own[u, i] - lr.cf[u, i, h] for u in range(t) if lr.assign[u, i] == g), 0.0) / (t * mu) for h in range(m)])
            q[g] = 1.0 - q.sum()
            assert lr.Q[t, i] == pytest.approx(q) and q.min() >= -1e-12
    cecurve, ccecurve = learn.gap_curves(inst, lr)
    for t in (0, T // 2, T - 1):
        tt = t + 1
        paid = [sum(lr.own[u, i] for u in range(tt)) for i in range(n)]
        best_ce = max(sum(lr.own[u, i] - lr.cf[u, i, h] for u in range(tt) if lr.assign[u, i] == g) for i in range(n) for g in range(m) for h in range(m))
        best_cce = max(paid[i] - min(sum(lr.cf[u, i, h] for u in range(tt)) for h in range(m)) for i in range(n))
        scale = float(np.mean(paid))
        assert cecurve[t] == pytest.approx(max(best_ce, 0.0) / scale) and ccecurve[t] == pytest.approx(best_cce / scale)


def test_hedge_equals_a_log_domain_loop_without_a_weight_floor():
    """Hedge per Schleife in der Log-Domäne mit gleichem Zufallsstrom. Eine Untergrenze der Gewichte (früher 1e-12) ließ abgeschlagene Tore zurückkehren: auf dieser Instanz wich die Ziehung ab Tag 661 ab."""
    inst = gates.generate(8, 3, "mixed", 0)
    n, m, T = inst.n, inst.m, 1000
    rng = np.random.default_rng(0)
    cmax = max(inst.a[h] + inst.b[h] * inst.w.sum() for h in range(m))
    S = [[0.0] * m for _ in range(n)]
    for t in range(T):
        P = []
        for i in range(n):
            top = max(S[i])
            e = [math.exp(s - top) for s in S[i]]
            P.append([x / sum(e) for x in e])
        u = rng.random(n)
        g = [min(sum(1 for h in range(m) if u[i] > sum(P[i][: h + 1])), m - 1) for i in range(n)]
        if t == 0:
            lr = learn.run_learning(inst, "hedge", T, 0)
        assert list(lr.assign[t]) == g, f"Tag {t}"
        assert lr.Q[t] == pytest.approx(np.array(P), abs=1e-9)
        for i in range(n):
            for h in range(m):
                S[i][h] -= C.HEDGE_ETA * _wait(inst, g, i, h) / cmax
    assert lr.Q.min() < 1e-12


@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 2.5, 4.0])
def test_mini_game_equals_hand_formulas_and_grid_search(delta):
    k1, k2 = bm.cost_matrices(delta)
    ag = (C.MINI_A, C.MINI_A + delta)
    for g1, g2 in itertools.product(range(2), repeat=2):
        same = 2 if g1 == g2 else 1
        assert k1[g1, g2] == pytest.approx(ag[g1] + C.MINI_B * same) and k2[g1, g2] == pytest.approx(ag[g2] + C.MINI_B * same)
    assert bm.pure_equilibria(k1, k2) == [(a, b) for a in range(2) for b in range(2) if k1[a, b] <= min(k1[:, b]) + 1e-9 and k2[a, b] <= min(k2[a, :]) + 1e-9]
    mixed = bm.mixed_equilibrium(k1, k2)
    if mixed is not None:
        p, q = mixed
        assert q * k1[0, 0] + (1 - q) * k1[0, 1] == pytest.approx(q * k1[1, 0] + (1 - q) * k1[1, 1])
        assert p * k2[0, 0] + (1 - p) * k2[1, 0] == pytest.approx(p * k2[0, 1] + (1 - p) * k2[1, 1])
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rows = []
    for i, km in enumerate((k1, k2)):
        for g in range(2):
            rows.append([0.0 if s[i] != g else km[s] - km[(1 - g, s[1]) if i == 0 else (s[0], 1 - g)] for s in states])
    rows = np.array(rows)
    c1, c2 = np.array([k1[s] for s in states]), np.array([k2[s] for s in states])
    x = bm.mediator(k1, k2, "welfare")
    ref = linprog(c1 + c2, A_ub=rows, b_ub=np.zeros(4), A_eq=np.ones((1, 4)), b_eq=[1.0], bounds=(0, None), method="highs-ipm")
    assert (rows @ x).max() < 1e-7 and (c1 + c2) @ x == pytest.approx(ref.fun)
    fair, grid, best = bm.mediator(k1, k2, "fair"), np.linspace(0, 1, 21), 1e9
    for a, b, c in itertools.product(grid, repeat=3):
        y = np.array([a, b, c, 1 - a - b - c])
        if y[3] >= -1e-12 and (rows @ y).max() <= 1e-9:
            best = min(best, max(c1 @ y, c2 @ y))
    assert (rows @ fair).max() < 1e-7 and max(c1 @ fair, c2 @ fair) <= best + 1e-9
