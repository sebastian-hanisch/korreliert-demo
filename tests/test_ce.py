"""Korrelierte Gleichgewichte per LP: Handrechnung im 2x2-Fall, Gehorsam unabhängig nachgerechnet, Inklusionen Nash ⊆ CE ⊆ CCE, LP-Lösung gegen ein zweites Lösungsverfahren."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

import kor_bimatrix as B
import kor_ce as CE
import kor_gates as G


def two_by_two(delta):
    """Zwei Lkw der Größe 1, Tore A (Grundzeit 5) und B (5 + delta), Zuschlag 2 je Lkw: dasselbe Spiel wie das Mini-Spiel."""
    return G.Instance(np.array([5.0, 5.0 + delta]), np.array([2.0, 2.0]), np.array([1.0, 1.0]), 0)


def cost_by_definition(inst, assign, i):
    load = sum(inst.w[j] for j in range(inst.n) if assign[j] == assign[i])
    return inst.a[assign[i]] + inst.b[assign[i]] * load


def deviation_by_definition(inst, assign, i, h):
    moved = list(assign)
    moved[i] = h
    return cost_by_definition(inst, moved, i)


def obedience_slack(inst, sigma):
    """Größte Verletzung der Gehorsamsbedingung, direkt aus der Definition über alle Zuordnungen (Zuordnung k = Ziffern zur Basis m, Lkw 0 niedrigstwertig)."""
    worst = -1e18
    states = list(itertools.product(range(inst.m), repeat=inst.n))
    for i in range(inst.n):
        for g in range(inst.m):
            for h in range(inst.m):
                if g == h:
                    continue
                total = 0.0
                for k in range(inst.m ** inst.n):
                    s = [(k // inst.m ** j) % inst.m for j in range(inst.n)]
                    if s[i] == g:
                        total += sigma[k] * (cost_by_definition(inst, s, i) - deviation_by_definition(inst, s, i, h))
                worst = max(worst, total)
    return worst


def test_states_agree_with_the_definition():
    inst = G.generate(4, 3, "mixed", 3)
    st = CE.States(inst)
    for k in range(st.N):
        s = [(k // inst.m ** j) % inst.m for j in range(inst.n)]
        assert list(st.A[k]) == s
        for i in range(inst.n):
            assert st.own[k, i] == pytest.approx(cost_by_definition(inst, s, i))
            for h in range(inst.m):
                assert st.dev[k, i, h] == pytest.approx(deviation_by_definition(inst, s, i, h))
        assert st.social[k] == pytest.approx(sum(cost_by_definition(inst, s, i) for i in range(inst.n)))


def test_too_many_assignments_raise():
    with pytest.raises(ValueError):
        CE.States(G.generate(11, 3, "mixed", 0))


def test_hand_calculated_two_by_two_game_at_delta_one():
    """delta = 1: reine Gleichgewichte (A,B) mit Kosten (7, 8) und (B,A) mit (8, 7); bester Vermittler = Summe 15 (ein reines Gleichgewicht), schlechtester per LP, nie unter 15."""
    st = CE.States(two_by_two(1.0))
    best, sigma = CE.solve(st, "ce", "min")
    assert best == pytest.approx(15.0)
    worst, _ = CE.solve(st, "ce", "max")
    assert worst >= 15.0 - 1e-9
    assert CE.solve(st, "cce", "min")[0] == pytest.approx(15.0)


def test_mediator_solutions_are_obedient_by_the_definition():
    for delta in (0.0, 1.0):
        inst = two_by_two(delta)
        st = CE.States(inst)
        for kind, sense in (("ce", "min"), ("ce", "max")):
            _, sigma = CE.solve(st, kind, sense)
            assert sigma.sum() == pytest.approx(1.0) and (sigma >= 0).all()
            assert obedience_slack(inst, sigma) <= 1e-7
    inst = G.generate(3, 3, "mixed", 5)
    st = CE.States(inst)
    _, sigma = CE.solve(st, "ce", "min")
    assert obedience_slack(inst, sigma) <= 1e-7


def test_ce_gap_of_the_lp_solutions_is_zero_and_of_a_bad_distribution_is_positive():
    inst = G.generate(4, 3, "mixed", 2)
    st = CE.States(inst)
    for sense in ("min", "max"):
        _, sigma = CE.solve(st, "ce", sense)
        assert CE.ce_gap(st, sigma) == pytest.approx((0.0, 0.0), abs=1e-7) or max(CE.ce_gap(st, sigma)) <= 1e-7
    bad = np.zeros(st.N)
    bad[int(np.argmax(st.social))] = 1.0                       # die schlechteste Zuordnung mit Sicherheit
    assert CE.ce_gap(st, bad)[0] > 0.1


@pytest.mark.parametrize("seed", range(6))
def test_inclusions_nash_subset_ce_subset_cce(seed):
    inst = G.generate(5, 3, "mixed", seed)
    st = CE.States(inst)
    en = G.analyse(inst)
    ce_min, ce_max = CE.solve(st, "ce", "min")[0], CE.solve(st, "ce", "max")[0]
    cce_min, cce_max = CE.solve(st, "cce", "min")[0], CE.solve(st, "cce", "max")[0]
    assert en["opt_cost"] - 1e-9 <= cce_min <= ce_min <= en["ne_cost_min"] + 1e-9
    assert en["ne_cost_max"] - 1e-9 <= ce_max <= cce_max + 1e-9


def test_every_pure_equilibrium_is_a_correlated_equilibrium():
    inst = G.generate(5, 3, "mixed", 4)
    st = CE.States(inst)
    en = G.analyse(inst)
    for row in en["ne_assignments"][:10]:
        k = int(sum(int(row[j]) * inst.m ** j for j in range(inst.n)))
        sigma = np.zeros(st.N)
        sigma[k] = 1.0
        assert max(CE.ce_gap(st, sigma)) <= 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_lp_optimum_agrees_with_a_second_solver(seed):
    inst = G.generate(5, 3, "mixed", seed)
    st = CE.States(inst)
    A = CE._obedience(st)
    for c in (st.social, -st.social):
        res = linprog(c, A_ub=A, b_ub=np.zeros(A.shape[0]), A_eq=np.ones((1, st.N)), b_eq=[1.0], bounds=(0, None), method="highs-ipm")
        assert res.success
        ours = CE.solve(st, "ce", "min" if c is st.social else "max")[0]
        assert ours == pytest.approx(float(st.social @ np.maximum(res.x, 0) / np.maximum(res.x, 0).sum()), rel=1e-6)


def test_two_by_two_lp_agrees_with_the_bimatrix_module():
    for delta in (0.0, 0.5, 1.0, 1.5, 3.0):
        K1, K2 = B.cost_matrices(delta)
        st = CE.States(two_by_two(delta))
        assert CE.solve(st, "ce", "min")[0] == pytest.approx(float((B.mediator(K1, K2, "welfare").reshape(2, 2) * (K1 + K2)).sum()), abs=1e-7)
        assert CE.solve(st, "ce", "max")[0] == pytest.approx(float((B.mediator(K1, K2, "worst").reshape(2, 2) * (K1 + K2)).sum()), abs=1e-7)


def test_support_of_a_vertex_solution_is_small():
    inst = G.generate(6, 3, "mixed", 35)
    st = CE.States(inst)
    _, sigma = CE.solve(st, "ce", "min")
    n_constraints = inst.n * inst.m * (inst.m - 1)
    assert 1 <= int((sigma > 1e-9).sum()) <= n_constraints + 1
