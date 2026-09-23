"""Mini-Spiel: Handrechnung (Kostenmatrizen, reine und gemischtes Gleichgewicht, gerechte Ampel) und Gehorsam der LP-Verteilungen."""

import numpy as np
import pytest

import kor_bimatrix as B


def test_cost_matrices_by_hand_at_delta_one():
    K1, K2 = B.cost_matrices(1.0)
    assert K1.tolist() == [[9.0, 7.0], [8.0, 10.0]]         # Zeile = Tor Lkw 1, Spalte = Tor Lkw 2
    assert K2.tolist() == [[9.0, 8.0], [7.0, 10.0]]


def test_pure_and_mixed_equilibria_by_hand():
    K1, K2 = B.cost_matrices(1.0)
    assert B.pure_equilibria(K1, K2) == [(0, 1), (1, 0)]
    p, q = B.mixed_equilibrium(K1, K2)
    assert p == pytest.approx(0.75) and q == pytest.approx(0.75)
    sigma = np.outer([p, 1 - p], [q, 1 - q])
    assert B.expected_costs(K1, K2, sigma) == pytest.approx((8.5, 8.5))       # 0,75 * 9 + 0,25 * 7 für Lkw 1 und dasselbe für Lkw 2


def test_mixed_equilibrium_makes_the_other_truck_indifferent():
    for delta in (0.0, 0.5, 1.0, 1.5):
        K1, K2 = B.cost_matrices(delta)
        p, q = B.mixed_equilibrium(K1, K2)
        assert q * K1[0, 0] + (1 - q) * K1[0, 1] == pytest.approx(q * K1[1, 0] + (1 - q) * K1[1, 1])
        assert p * K2[0, 0] + (1 - p) * K2[1, 0] == pytest.approx(p * K2[0, 1] + (1 - p) * K2[1, 1])


def test_dominant_gate_for_delta_two_or_more():
    for delta in (2.5, 3.0, 4.0):
        K1, K2 = B.cost_matrices(delta)
        assert B.pure_equilibria(K1, K2) == [(0, 0)] and B.mixed_equilibrium(K1, K2) is None


def test_fair_ampel_alternates_the_two_pure_equilibria_by_hand():
    K1, K2 = B.cost_matrices(1.0)
    sigma = B.mediator(K1, K2, "fair")
    assert sigma.tolist() == pytest.approx([0.0, 0.5, 0.5, 0.0])
    assert B.expected_costs(K1, K2, sigma) == pytest.approx((7.5, 7.5))       # 0,5 * 7 + 0,5 * 8 für jeden Lkw: besser als 8,5 im gemischten Gleichgewicht


def test_ampel_beats_the_mixed_equilibrium_for_both_trucks_below_delta_two():
    for delta in (0.0, 0.5, 1.0, 1.5):
        K1, K2 = B.cost_matrices(delta)
        p, q = B.mixed_equilibrium(K1, K2)
        mixed = B.expected_costs(K1, K2, np.outer([p, 1 - p], [q, 1 - q]))
        fair = B.expected_costs(K1, K2, B.mediator(K1, K2, "fair"))
        assert fair[0] < mixed[0] - 1e-9 and fair[1] < mixed[1] - 1e-9 and fair[0] == pytest.approx(fair[1])


@pytest.mark.parametrize("objective", ["welfare", "fair", "worst"])
@pytest.mark.parametrize("delta", [0.0, 1.0, 2.0, 3.5])
def test_mediator_distributions_are_obedient(objective, delta):
    K1, K2 = B.cost_matrices(delta)
    sigma = B.mediator(K1, K2, objective)
    assert sigma.sum() == pytest.approx(1.0) and (sigma >= -1e-12).all()
    assert (B.obedience_rows(K1, K2) @ sigma <= 1e-8).all()


def test_worst_mediator_is_not_better_than_welfare_mediator_and_unknown_objective_raises():
    K1, K2 = B.cost_matrices(1.0)
    welfare = float((B.mediator(K1, K2, "welfare").reshape(2, 2) * (K1 + K2)).sum())
    worst = float((B.mediator(K1, K2, "worst").reshape(2, 2) * (K1 + K2)).sum())
    assert worst >= welfare - 1e-9
    with pytest.raises(ValueError):
        B.mediator(K1, K2, "bogus")
