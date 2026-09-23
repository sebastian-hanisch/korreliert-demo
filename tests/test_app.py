"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Empfehlungs-Check, Würfel-Knopf, Permalink-Grenzen, Extremwerte, Mini-Spiel, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kor_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_has_no_exception_and_shows_the_mediator_success():
    at = _run()
    _ok(at)
    assert at.metric and any("schlägt jedes reine Gleichgewicht" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["n_slider"] == p["n"] and at.session_state["m_slider"] == p["m"]
    assert at.metric


def test_no_gain_verdict_for_two_gates():
    at = _run(m_slider=2)
    _ok(at)
    assert any("gewinnt der Vermittler nichts" in i.value for i in at.info)


def test_recommendation_check_runs_for_every_truck_and_gate():
    at = _run(n_slider=4, m_slider=3)
    _ok(at)
    for truck in (1, 4):
        for gate in (1, 2, 3):
            at.session_state["check_truck"] = truck
            at.session_state["check_gate"] = gate
            at.run()
            _ok(at)
    at.radio(key="check_kind").set_value("ce_max").run()
    _ok(at)


def test_recommendation_check_clamps_after_shrinking_the_instance():
    at = _run(n_slider=8, m_slider=3, check_truck=8, check_gate=3)
    _ok(at)
    at.session_state["n_slider"] = 4
    at.session_state["m_slider"] = 2
    at.run()
    _ok(at)
    assert at.session_state["check_truck"] <= 4 and at.session_state["check_gate"] <= 2


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Vehikel generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["delta"] = "0.7"
    at.query_params["sizes"] = "uniform"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["size_select"] == "uniform"
    assert abs(at.session_state["delta_slider"] - 0.5) < 1e-9 or abs(at.session_state["delta_slider"] - 1.0) < 1e-9


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN), dict(m_slider=C.M_MIN), dict(size_select="uniform"), dict(delta_slider=C.MINI_DELTA_MIN), dict(delta_slider=C.MINI_DELTA_MAX), dict(delta_slider=2.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**{"n_slider": 5, **kw}))


def test_mini_game_shows_the_ampel_and_the_dominant_case():
    at = _run(delta_slider=1.0)
    assert any("Gerechte Ampel" in m.value for m in at.markdown)
    at = _run(delta_slider=3.0)
    _ok(at)
    assert any("Ab δ ≥ 2" in c.value for c in at.caption)


def test_mediator_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "MEDIATOR_SEEDS", tuple(range(800000, 800004)))
    at = _run()
    next(b for b in at.button if b.key == "mediator_start").click().run()
    _ok(at)
    assert at.session_state["mediator_on"] and at.get("plotly_chart")


def test_scaling_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "SCALING_NS", (4, 5))
    monkeypatch.setattr(C, "SCALING_SEEDS", tuple(range(810000, 810004)))
    at = _run()
    next(b for b in at.button if b.key == "scaling_start").click().run()
    _ok(at)
    assert at.session_state["scaling_on"]


def test_learning_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "LEARN_EXP_SEEDS", tuple(range(820000, 820003)))
    monkeypatch.setattr(C, "LEARN_EXP_T", 200)
    at = _run()
    next(b for b in at.button if b.key == "learning_start").click().run()
    _ok(at)
    assert at.session_state["learning_on"]


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
