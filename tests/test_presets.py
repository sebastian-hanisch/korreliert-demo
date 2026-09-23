"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft)."""

import kor_constants as C
import kor_evaluation as E
import kor_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name] and "TODO" not in C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for p in C.PRESETS.values():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and p["size_mode"] in C.SIZE_MODES
        assert C.MINI_DELTA_MIN <= p["delta"] <= C.MINI_DELTA_MAX
        assert p["m"] ** p["n"] <= C.ENUM_MAX_ASSIGNMENTS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall (8 Lkw, 3 Tore)"]
    assert E.Settings(p["n"], p["m"], p["size_mode"], p["seed"]) == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("delta_slider") == (C.MINI_DELTA_MIN, C.MINI_DELTA_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"n_slider", "m_slider", "delta_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_slider_limits_keep_the_lp_size_below_the_cap():
    assert C.M_MAX ** C.N_MAX <= C.ENUM_MAX_ASSIGNMENTS
