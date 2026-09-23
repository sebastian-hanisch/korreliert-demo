"""Konstanten der Korrelierte-Gleichgewichte-Demo: Vehikel "Torwahl" aus nash-demo, Vermittler (LP), Lernverfahren, Mini-Spiel, Experimente (Presets nach den Messungen)."""

EPS = 1e-9                         # ein Wechsel/eine Verbesserung zählt nur, wenn sie echt ist
SEED_MAX = 999999

# --- Vehikel "Torwahl" (wortgleich zu nash-demo) ----------------------------------------------------------------------------------------------

A_MIN, A_MAX = 2.0, 10.0           # Grundwartezeit a_g je Tor in Minuten
B_MIN, B_MAX = 0.5, 3.0            # Zuschlag b_g je Ladungseinheit in Minuten
SIZES = (1, 2, 3)                  # Lkw-Größen: Transporter, Lkw, Sattelzug
SIZE_PROBS = (0.5, 0.3, 0.2)
SIZE_MODES = ("mixed", "uniform")
SIZE_MODE_LABELS = {"mixed": "Gemischt (1/2/3)", "uniform": "Einheitlich (alle 1)"}
ENUM_MAX_ASSIGNMENTS = 100_000     # Das LP hat eine Variable je Zuordnung: m^n darf diese Grenze nicht übersteigen

N_MIN, N_MAX, DEFAULT_N, N_STEP = 4, 10, 8, 1
M_MIN, M_MAX, DEFAULT_M, M_STEP = 2, 3, 3, 1
DEFAULT_SEED = 35                  # Vehikel-Seed (wie nash-demo)

# --- Lernen (Vergleich Hedge gegen Regret-Matching) -------------------------------------------------------------------------------------------

LEARN_METHODS = ("rm", "hedge")
LEARN_LABELS = {"rm": "Regret-Matching (Hart/Mas-Colell)", "hedge": "Hedge (externer Regret)"}
HEDGE_ETA = 10.0                   # feste Schrittweite, wie im Standardfall von noregret-demo
RM_MU_FACTOR = 1.0                 # Normierung der Wechselwahrscheinlichkeit: mu = RM_MU_FACTOR * (m - 1) * größte mögliche Wartezeit
T_MIN, T_MAX, DEFAULT_T, T_STEP = 200, 5000, 3000, 200

# --- Mini-Spiel: zwei Lkw, zwei Tore (wie nash-demo) ------------------------------------------------------------------------------------------

MINI_A, MINI_B = 5.0, 2.0          # beide Tore: Grundzeit 5, Zuschlag 2 je Einheit; Tor B ist um DELTA langsamer
MINI_DELTA_MIN, MINI_DELTA_MAX, DEFAULT_MINI_DELTA, MINI_DELTA_STEP = 0.0, 4.0, 1.0, 0.5

# --- Experimente (feste Seeds) ---------------------------------------------------------------------------------------------------------------

MEDIATOR_N, MEDIATOR_M = 6, 3
MEDIATOR_SEEDS = tuple(range(800000, 800060))              # Vehikel-Seeds Experiment 1
SCALING_NS = (4, 5, 6, 7, 8)
SCALING_SEEDS = tuple(range(810000, 810030))
LEARN_EXP_SEEDS = tuple(range(820000, 820020))              # Experiment 3 (Lernen)
LEARN_EXP_N, LEARN_EXP_M = 8, 3
LEARN_EXP_T = 3000
CHECK_DAYS = (50, 100, 200, 500, 1000, 2000, 3000)

# --- Presets (Werte nach den Messungen) ----------------------------------------------------------------------------------------------------


def _preset(n=DEFAULT_N, m=DEFAULT_M, size_mode="mixed", seed=DEFAULT_SEED, delta=DEFAULT_MINI_DELTA):
    return {"n": n, "m": m, "size_mode": size_mode, "seed": seed, "delta": delta}


PRESETS = {
    "Standardfall (8 Lkw, 3 Tore)": _preset(),
    "Kleine Instanz (6 Lkw)": _preset(n=6),
    "Einheitliche Lkw": _preset(size_mode="uniform"),
    "Neun Lkw": _preset(n=9),
    "Zwei Tore (kein Gewinn)": _preset(m=2),
    "Anderes Vehikel (Seed 7)": _preset(seed=7),
}
PRESET_HELP = {
    "Standardfall (8 Lkw, 3 Tore)": "Optimum 95,5 min, reine Nash-Gleichgewichte 96,9 bis 100,3 min. Der beste Vermittler erreicht 96,1 min - besser als jedes reine Gleichgewicht - mit 9 verschiedenen empfohlenen Zuordnungen. Der schlechteste (108,0 min) ist dafür auch schlechter als jedes reine Gleichgewicht.",
    "Kleine Instanz (6 Lkw)": "Alle reinen Gleichgewichte liegen bei 68,9 min (Optimum 65,1). Der Vermittler kommt auf 66,1 min, das grobe korrelierte Gleichgewicht sogar auf 65,18 - fast das Optimum.",
    "Einheitliche Lkw": "Alle Lkw Größe 1: alle reinen Gleichgewichte liegen bei 78,3 min (Optimum 75,1). Der Vermittler erreicht 76,7 min, das grobe korrelierte Gleichgewicht 75,3.",
    "Neun Lkw": "Mit 9 Lkw schließt der Vermittler fast die ganze Lücke: 109,6 min gegenüber 112,2 im besten reinen Gleichgewicht und 109,3 im Optimum.",
    "Zwei Tore (kein Gewinn)": "Bei nur zwei Toren trifft schon ein reines Gleichgewicht das Optimum (217,3 min): der Vermittler hat nichts zu verbessern - nur zu verschlechtern (schlechtestes korreliertes Gleichgewicht 246,8).",
    "Anderes Vehikel (Seed 7)": "Ein anderes Vehikel: reine Gleichgewichte 105,9 bis 107,0 min, bester Vermittler 103,9, Optimum 103,0.",
}
