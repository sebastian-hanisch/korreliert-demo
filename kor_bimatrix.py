"""Mini-Spiel: zwei Lkw (Größe 1), zwei Tore mit gleicher Grundzeit; Tor B ist um delta langsamer. Kostenmatrizen, reine und gemischtes Gleichgewicht (wie nash-demo) und der Vermittler als
kleines LP über die vier Zuordnungen (Tor Lkw 1, Tor Lkw 2)."""

import numpy as np
from scipy.optimize import linprog

import kor_constants as C

STATES = ((0, 0), (0, 1), (1, 0), (1, 1))


def cost_matrices(delta, a=C.MINI_A, b=C.MINI_B):
    """(K1, K2): Kosten von Lkw 1/2, Zeile = Tor von Lkw 1, Spalte = Tor von Lkw 2 (0 = Tor A, 1 = Tor B)."""
    a_g = (a, a + delta)
    K1, K2 = np.zeros((2, 2)), np.zeros((2, 2))
    for g1 in range(2):
        for g2 in range(2):
            K1[g1, g2] = a_g[g1] + b * (1 + (g1 == g2))
            K2[g1, g2] = a_g[g2] + b * (1 + (g1 == g2))
    return K1, K2


def pure_equilibria(K1, K2):
    return [(g1, g2) for g1 in range(2) for g2 in range(2)
            if K1[g1, g2] <= K1[1 - g1, g2] + C.EPS and K2[g1, g2] <= K2[g1, 1 - g2] + C.EPS]


def mixed_equilibrium(K1, K2):
    """Vollständig gemischtes Gleichgewicht (p = Wahrscheinlichkeit, dass Lkw 1 Tor A wählt, q = dito Lkw 2) oder None (Indifferenzformeln)."""
    d1 = K1[0, 0] - K1[0, 1] - K1[1, 0] + K1[1, 1]
    d2 = K2[0, 0] - K2[1, 0] - K2[0, 1] + K2[1, 1]
    if abs(d1) < C.EPS or abs(d2) < C.EPS:
        return None
    q = (K1[1, 1] - K1[0, 1]) / d1
    p = (K2[1, 1] - K2[1, 0]) / d2
    return (float(p), float(q)) if 0.0 < p < 1.0 and 0.0 < q < 1.0 else None


def expected_costs(K1, K2, sigma):
    sigma = np.asarray(sigma).reshape(2, 2)
    return float((sigma * K1).sum()), float((sigma * K2).sum())


def obedience_rows(K1, K2):
    """Ungleichungen A sigma <= 0 des Vermittlers: Lkw 1 mit Empfehlung g wechselt nicht zu 1-g, Lkw 2 ebenso."""
    rows = []
    for g in range(2):
        r = np.zeros(4)                                            # Lkw 1, Empfehlung g
        for g2 in range(2):
            r[STATES.index((g, g2))] = K1[g, g2] - K1[1 - g, g2]
        rows.append(r)
    for g in range(2):
        r = np.zeros(4)                                            # Lkw 2, Empfehlung g
        for g1 in range(2):
            r[STATES.index((g1, g))] = K2[g1, g] - K2[g1, 1 - g]
        rows.append(r)
    return np.array(rows)


def mediator(K1, K2, objective="welfare"):
    """Vermittler als LP: 'welfare' = kleinste Summe der erwarteten Kosten, 'fair' = kleinste größere erwartete Kosten der beiden Lkw (gerecht),
    'worst' = größte Summe. Rückgabe: Verteilung über STATES."""
    A = obedience_rows(K1, K2)
    c1, c2 = K1.reshape(4), K2.reshape(4)
    if objective == "fair":
        cost = np.array([0, 0, 0, 0, 1.0])
        A_ub = np.vstack([np.hstack([A, np.zeros((4, 1))]), np.append(c1, -1.0), np.append(c2, -1.0)])
        A_eq = np.append(np.ones(4), 0.0)[None, :]
        res = linprog(cost, A_ub=A_ub, b_ub=np.zeros(6), A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * 5, method="highs")
        x = res.x[:4]
    elif objective in ("welfare", "worst"):
        sign = 1.0 if objective == "welfare" else -1.0
        res = linprog(sign * (c1 + c2), A_ub=A, b_ub=np.zeros(4), A_eq=np.ones((1, 4)), b_eq=[1.0], bounds=[(0, None)] * 4, method="highs")
        x = res.x
    else:
        raise ValueError(objective)
    x = np.maximum(x, 0.0)
    return x / x.sum()
