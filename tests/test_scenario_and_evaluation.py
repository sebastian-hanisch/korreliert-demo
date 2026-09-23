"""Auswertung: Analyse einer Instanz und die drei Experimente - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import kor_constants as C
import kor_evaluation as E


def test_analyse_default_is_consistent():
    a = E.analyse(E.Settings(n=6))
    r = a.ratios()
    assert r["opt"] == 1.0 and r["cce_min"] <= r["ce_min"] <= r["ne_min"] <= r["ne_max"] <= r["ce_max"] <= r["cce_max"] + 1e-9
    assert a.support_size("ce_min") >= 1
    top = a.top_assignments("ce_min", 5)
    assert all(p1 >= p2 - 1e-12 for (_, p1, _), (_, p2, _) in zip(top, top[1:])) and sum(p for _, p, _ in top) <= 1 + 1e-9


def test_follow_or_deviate_shows_obedience_for_every_recommendation():
    """Bei jeder ausgesprochenen Empfehlung ist Gehorsam mindestens so gut wie jedes andere feste Tor (Definition des korrelierten Gleichgewichts, bedingt auf die Empfehlung)."""
    a = E.analyse(E.Settings(n=6, m=3))
    for kind in ("ce_min", "ce_max"):
        for i in range(a.inst.n):
            for g in range(a.inst.m):
                vals, p = a.follow_or_deviate(kind, i, g)
                if vals is not None:
                    assert vals[g] <= vals.min() + 1e-7 and p > 0
    assert sum(a.follow_or_deviate("ce_min", 0, g)[1] for g in range(a.inst.m)) == pytest.approx(1.0)


def test_follow_or_deviate_probabilities_sum_to_one_over_recommendations_and_unused_ones_return_none():
    a = E.analyse(E.Settings(n=4, m=2))
    for i in range(4):
        total = 0.0
        for g in range(2):
            vals, p = a.follow_or_deviate("ce_min", i, g)
            assert (vals is None) == (p == 0.0)
            total += p
        assert total == pytest.approx(1.0)


def test_mediator_experiment_shape_and_invariants():
    r = E.mediator_experiment("mixed", n=5, seeds=range(800000, 800008))
    a = r["arrays"]
    assert r["n_inst"] == 8 and np.all(a["cce_min"] <= a["ce_min"] + 1e-9) and np.all(a["ce_min"] <= a["ne_min"] + 1e-9) and np.all(a["ne_max"] <= a["ce_max"] + 1e-9)
    assert 0 <= r["share_ce_beats_ne"] <= 1 and r["share_ce_optimal"] <= r["share_cce_optimal"] + 1e-12


def test_scaling_experiment_shape():
    rows = E.scaling_experiment(ns=(4, 5), seeds=range(810000, 810005))
    assert [r["n"] for r in rows] == [4, 5] and all(r["ce_min"] <= r["ne_min"] + 1e-9 for r in rows)


def test_learning_experiment_shape_and_invariants():
    r = E.learning_experiment(T=200, seeds=range(820000, 820003), days=(50, 200))
    assert r["days"] == (50, 200) and r["n_inst"] == 3
    for method in C.LEARN_METHODS:
        assert len(r[method]["ce_gap"]) == 2 and r[method]["window"] >= 1 - 1e-9 and 0 <= r[method]["last_is_ne"] <= 1


def test_experiments_are_reproducible():
    a = E.learning_experiment(T=100, seeds=range(820000, 820002), days=(100,))
    b = E.learning_experiment(T=100, seeds=range(820000, 820002), days=(100,))
    assert np.array_equal(a["rm"]["ce_gap"], b["rm"]["ce_gap"]) and a["hedge"]["window"] == b["hedge"]["window"]
