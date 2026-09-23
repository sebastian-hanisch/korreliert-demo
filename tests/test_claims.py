"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Die LP-Werte (bester/schlechtester Vermittler) sind eindeutige Optimalwerte einer festen Instanz und damit plattformrobust; Mittel über Instanzen bekommen enge, aber nicht kleinliche Bänder.
Das Lernen ist ein Zufallsprozess mit festen Seeds: dort nur großzügige Bänder (feedback_ci_platform_robust_tests / feedback_ci_unpinned_numeric_asserts)."""

import numpy as np
import pytest

import kor_bimatrix as B
import kor_constants as C
import kor_evaluation as E


def _preset(name):
    p = C.PRESETS[name]
    return E.analyse(E.Settings(p["n"], p["m"], p["size_mode"], p["seed"]))


# --- PRESET_HELP: exakte LP-Werte ------------------------------------------------------------------------------------------------------------


def _vals(a):
    return (a.opt, a.enum["ne_cost_min"], a.enum["ne_cost_max"], a.ce_min[0], a.ce_max[0], a.cce_min[0])


def test_preset_numbers():
    expected = {
        "Standardfall (8 Lkw, 3 Tore)": (95.46, 96.90, 100.32, 96.12, 108.04, 96.12),
        "Kleine Instanz (6 Lkw)": (65.14, 68.90, 68.90, 66.11, 70.61, 65.18),
        "Einheitliche Lkw": (75.07, 78.27, 78.27, 76.72, 82.77, 75.30),
        "Neun Lkw": (109.29, 112.21, 115.32, 109.58, 123.56, 109.58),
        "Zwei Tore (kein Gewinn)": (217.25, 217.25, 224.03, 217.25, 246.83, 217.25),
        "Anderes Vehikel (Seed 7)": (103.02, 105.88, 106.97, 103.87, 115.55, 103.40),
    }
    for name, exp in expected.items():
        assert _vals(_preset(name)) == pytest.approx(exp, abs=0.006), name


def test_standardfall_structure():
    a = _preset("Standardfall (8 Lkw, 3 Tore)")
    assert a.enum["n_ne"] == 80 and a.support_size("ce_min") == 9
    assert a.ce_min[0] < a.enum["ne_cost_min"] - 0.5 and a.ce_max[0] > a.enum["ne_cost_max"] + 5
    gap_closed = (a.enum["ne_cost_min"] - a.ce_min[0]) / (a.enum["ne_cost_min"] - a.opt)
    assert gap_closed == pytest.approx(0.54, abs=0.01)


def test_zwei_tore_no_gain_and_kleine_instanz_cce_near_optimum():
    two = _preset("Zwei Tore (kein Gewinn)")
    assert two.ce_min[0] == pytest.approx(two.opt) and two.enum["ne_cost_min"] == pytest.approx(two.opt)
    small = _preset("Kleine Instanz (6 Lkw)")
    assert small.cce_min[0] - small.opt < 0.05 and small.cce_min[0] < small.ce_min[0] - 0.5


def test_neun_lkw_closes_most_of_the_gap():
    a = _preset("Neun Lkw")
    assert (a.enum["ne_cost_min"] - a.ce_min[0]) / (a.enum["ne_cost_min"] - a.opt) > 0.85


# --- Experiment 1: Vermittler (60 Instanzen, 6 Lkw) --------------------------------------------------------------------------------------------


def test_mediator_experiment_numbers():
    m, u = E.mediator_experiment("mixed"), E.mediator_experiment("uniform")
    assert m["n_inst"] == 60
    for key, val in (("ne_min_mean", 1.0253), ("ce_min_mean", 1.0149), ("cce_min_mean", 1.0106), ("ne_max_mean", 1.0667), ("ce_max_mean", 1.1896), ("cce_max_mean", 1.1920)):
        assert m[key] == pytest.approx(val, abs=0.002), key
    assert m["share_ce_beats_ne"] == pytest.approx(0.50, abs=0.03) and m["share_cce_beats_ce"] == pytest.approx(0.383, abs=0.03) and m["share_ce_optimal"] == pytest.approx(0.35, abs=0.03)
    assert m["ce_max_max"] == pytest.approx(1.299, abs=0.003)
    for key, val in (("ne_min_mean", 1.0214), ("ce_min_mean", 1.0128), ("cce_min_mean", 1.0059), ("ne_max_mean", 1.0214), ("ce_max_mean", 1.1192)):
        assert u[key] == pytest.approx(val, abs=0.002), key
    assert m["ce_min_mean"] < m["ne_min_mean"] and m["ce_max_mean"] > m["ne_max_mean"] + 0.1 and u["ne_min_mean"] == pytest.approx(u["ne_max_mean"])


# --- Experiment 2: Skalierung ------------------------------------------------------------------------------------------------------------------


def test_scaling_experiment_numbers():
    rows = {r["n"]: r for r in E.scaling_experiment()}
    assert sorted(rows) == [4, 5, 6, 7, 8]
    shares = [rows[n]["share_ce_beats_ne"] for n in (4, 5, 6, 7, 8)]
    assert shares[0] < 0.15 and shares[-1] > 0.6 and shares[-1] == pytest.approx(0.733, abs=0.04) and shares[0] == pytest.approx(0.067, abs=0.04)
    assert rows[4]["ne_min"] == pytest.approx(1.0072, abs=0.002) and rows[8]["ne_min"] == pytest.approx(1.0294, abs=0.002) and rows[8]["ce_min"] == pytest.approx(1.0178, abs=0.002)
    assert all(rows[n]["ce_min"] <= rows[n]["ne_min"] + 1e-9 for n in rows) and all(rows[n]["cce_min"] <= rows[n]["ce_min"] + 1e-9 for n in rows)


# --- Experiment 3: Lernen (Bänder) -------------------------------------------------------------------------------------------------------------


def test_learning_experiment_numbers():
    r = E.learning_experiment()
    rm, hd = r["rm"], r["hedge"]
    assert r["days"][-1] == 3000 and r["ref_ne_min"] == pytest.approx(1.0255, abs=0.003) and r["ref_ce_min"] == pytest.approx(1.0182, abs=0.003)
    assert 0.001 < rm["ce_gap"][-1] < 0.004 and 0.0002 < hd["ce_gap"][-1] < 0.0015 and hd["ce_gap"][-1] < rm["ce_gap"][-1]
    assert rm["ce_gap"][0] > 0.05 and hd["ce_gap"][0] > 0.02 and rm["ce_gap"][-1] < rm["ce_gap"][0] / 10
    assert 1.03 < rm["window"] < 1.06 and 1.03 < hd["window"] < 1.06 and rm["window"] > r["ref_ce_min"] + 0.015 and hd["window"] > r["ref_ce_min"] + 0.015
    assert rm["last_is_ne"] >= 0.85 and hd["last_is_ne"] >= 0.95


# --- Mini-Spiel --------------------------------------------------------------------------------------------------------------------------------


def test_mini_game_numbers_at_the_default_delta():
    K1, K2 = B.cost_matrices(C.DEFAULT_MINI_DELTA)
    assert B.pure_equilibria(K1, K2) == [(0, 1), (1, 0)] and (K1[0, 1], K2[0, 1]) == (7.0, 8.0)
    p, q = B.mixed_equilibrium(K1, K2)
    assert (p, q) == (pytest.approx(0.75), pytest.approx(0.75))
    assert B.expected_costs(K1, K2, np.outer([p, 1 - p], [q, 1 - q])) == pytest.approx((8.5, 8.5))
    assert B.expected_costs(K1, K2, B.mediator(K1, K2, "fair")) == pytest.approx((7.5, 7.5))
