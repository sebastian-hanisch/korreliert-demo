"""Lernen: Regret-Matching-Wahrscheinlichkeiten gegen eine unabhängige Schleife, CE-Lücke gegen die LP-Auswertung derselben Verteilung, Hedge, Reproduzierbarkeit."""

import numpy as np
import pytest

import kor_ce as CE
import kor_constants as C
import kor_gates as G
import kor_learning as L


def test_first_day_probabilities_are_uniform_and_rows_are_distributions():
    inst = G.generate(5, 3, "mixed", 1)
    for method in C.LEARN_METHODS:
        lr = L.run_learning(inst, method, 50, seed=2)
        assert np.allclose(lr.Q[0], 1 / 3) and np.allclose(lr.Q.sum(axis=2), 1.0) and (lr.Q >= -1e-12).all()


def test_regret_matching_probabilities_agree_with_an_independent_loop():
    inst = G.generate(5, 3, "mixed", 4)
    lr = L.run_learning(inst, "rm", 80, seed=3)
    n, m = inst.n, inst.m
    mu = C.RM_MU_FACTOR * (m - 1) * L.normalizer(inst)
    D = np.zeros((n, m, m))
    for t in range(1, lr.T):
        g_prev = lr.assign[t - 1]
        for i in range(n):
            for h in range(m):
                D[i, g_prev[i], h] += lr.own[t - 1, i] - lr.cf[t - 1, i, h]
        for i in range(n):
            expected = np.zeros(m)
            for h in range(m):
                if h != g_prev[i]:
                    expected[h] = max(D[i, g_prev[i], h], 0.0) / (t * mu)
            expected[g_prev[i]] = 1.0 - expected.sum()
            assert lr.Q[t, i] == pytest.approx(expected, abs=1e-12)
        # der gezogene Tor hatte positive Wahrscheinlichkeit
        assert all(lr.Q[t, i, lr.assign[t, i]] > 0 for i in range(n))


def test_an_agent_that_never_regrets_stays_put():
    """Konstante Zuordnung auf einem Nash-Gleichgewicht: kein positiver Regret, also bleibt jeder bei seinem Tor (Regret-Matching-Übergang)."""
    inst = G.generate(5, 3, "mixed", 2)
    en = G.analyse(inst)
    ne = en["ne_assignments"][0].astype(int)
    T = 30
    assign = np.tile(ne, (T, 1))
    cost = L.cost_matrix(inst, ne)
    own = np.tile(cost[np.arange(inst.n), ne], (T, 1))
    cf = np.tile(cost, (T, 1, 1))
    lr = L.Learning("rm", T, assign, own, cf, np.zeros(T, dtype=int), np.zeros((T, inst.n, inst.m)))
    ce, cce = L.gap_curves(inst, lr)
    assert np.allclose(ce, 0.0, atol=1e-12) and np.all(cce <= 1e-12)


def test_gap_curves_agree_with_the_lp_evaluation_of_the_empirical_distribution():
    """CE- und CCE-Lücke der empirischen Verteilung nach T Tagen: unabhängig berechnet einmal über die Summen der Lernschleife, einmal über kor_ce.ce_gap auf der Verteilung."""
    inst = G.generate(5, 3, "mixed", 6)
    st = CE.States(inst)
    for method in C.LEARN_METHODS:
        lr = L.run_learning(inst, method, 400, seed=1)
        sigma = np.zeros(st.N)
        for t in range(lr.T):
            sigma[int(sum(int(lr.assign[t, j]) * inst.m ** j for j in range(inst.n)))] += 1.0 / lr.T
        ce, cce = L.gap_curves(inst, lr)
        scale = np.cumsum(lr.own, axis=0).mean(axis=1)[-1]
        gap_ce, gap_cce = CE.ce_gap(st, sigma)
        assert ce[-1] == pytest.approx(max(gap_ce, 0.0) * lr.T / scale, rel=1e-6, abs=1e-12)
        assert cce[-1] == pytest.approx(max(gap_cce, 0.0) * lr.T / scale, rel=1e-6, abs=1e-12) or cce[-1] <= 0


def test_gap_curves_agree_with_an_independent_loop():
    inst = G.generate(4, 3, "mixed", 8)
    lr = L.run_learning(inst, "hedge", 60, seed=5)
    ce, cce = L.gap_curves(inst, lr)
    R = np.zeros((inst.n, inst.m, inst.m))
    paid = np.zeros(inst.n)
    fixed = np.zeros((inst.n, inst.m))
    for t in range(lr.T):
        for i in range(inst.n):
            R[i, lr.assign[t, i], :] += lr.own[t, i] - lr.cf[t, i]
        paid += lr.own[t]
        fixed += lr.cf[t]
        assert ce[t] == pytest.approx(max(R.max(), 0.0) / paid.mean(), abs=1e-9)
        assert cce[t] == pytest.approx((paid - fixed.min(axis=1)).max() / paid.mean(), abs=1e-9)


def test_both_methods_bring_the_ce_gap_down_on_the_default_instance():
    inst = G.generate(8, 3, "mixed", 35)
    for method in C.LEARN_METHODS:
        lr = L.run_learning(inst, method, 2000, seed=7)
        ce, _ = L.gap_curves(inst, lr)
        assert ce[-1] < 0.02 and ce[-1] < ce[99]


def test_reproducible_and_seed_dependent_and_unknown_method_raises():
    inst = G.generate(6, 3, "mixed", 2)
    a, b, c = (L.run_learning(inst, "rm", 100, seed=s) for s in (1, 1, 2))
    assert np.array_equal(a.assign, b.assign) and not np.array_equal(a.assign, c.assign)
    with pytest.raises(ValueError):
        L.run_learning(inst, "bogus", 10)
