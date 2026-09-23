"""Torwahl-Vehikel: wortgleich zu nash-demo (Zahlen von dort), Aufzählung gegen eine skalare Definition."""

import itertools

import numpy as np
import pytest

import kor_gates as G


def cost_of(inst, assign, i):
    loads = np.bincount(assign, weights=inst.w, minlength=inst.m)
    return inst.a[assign[i]] + inst.b[assign[i]] * loads[assign[i]]


def is_ne_by_definition(inst, assign):
    for i in range(inst.n):
        cur = cost_of(inst, assign, i)
        for h in range(inst.m):
            moved = assign.copy()
            moved[i] = h
            if cost_of(inst, moved, i) < cur - 1e-9:
                return False
    return True


def test_matches_nash_demo_on_the_default_instances():
    """Zahlen aus nash-demo (Seed 35, 8 Lkw, 3 Tore, gemischt): 80 Gleichgewichte (96,9 bis 100,3 min), Optimum 95,5."""
    r = G.analyse(G.generate(8, 3, "mixed", 35))
    assert r["n_ne"] == 80 and abs(r["opt_cost"] - 95.5) < 0.05 and abs(r["ne_cost_min"] - 96.9) < 0.05 and abs(r["ne_cost_max"] - 100.3) < 0.05


def test_uniform_mode_has_the_same_gates_as_mixed_and_errors():
    mixed, uniform = G.generate(12, 3, "mixed", 9), G.generate(12, 3, "uniform", 9)
    assert np.array_equal(mixed.a, uniform.a) and np.array_equal(mixed.b, uniform.b) and np.all(uniform.w == 1.0)
    with pytest.raises(ValueError):
        G.generate(4, 2, "bogus", 0)
    with pytest.raises(ValueError):
        G.analyse(G.generate(12, 4, "mixed", 0))


@pytest.mark.parametrize("seed", range(5))
def test_enumeration_agrees_with_the_scalar_definition(seed):
    inst = G.generate(6, 3, "mixed", seed)
    res = G.analyse(inst)
    states = [np.array(a) for a in itertools.product(range(3), repeat=6)]
    ne = [G.social_cost(inst, a) for a in states if is_ne_by_definition(inst, a)]
    assert res["n_ne"] == len(ne) and res["ne_cost_min"] == pytest.approx(min(ne)) and res["ne_cost_max"] == pytest.approx(max(ne))
    assert res["opt_cost"] == pytest.approx(min(G.social_cost(inst, a) for a in states))
