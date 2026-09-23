"""Korrelierte Gleichgewichte des Torwahl-Spiels per linearem Programm (Aumann 1974): ein Vermittler zieht eine Zuordnung s aus einer Verteilung sigma und teilt jedem Lkw nur sein Tor mit.
Gehorsam: kein Lkw verbessert sich, wenn er die Empfehlung g durch ein festes anderes Tor h ersetzt. Grobes korreliertes Gleichgewicht (CCE): kein Lkw verbessert sich durch ein festes anderes Tor,
bevor er seine Empfehlung kennt."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import kor_constants as C
import kor_gates as G


class States:
    """Alle m^n Zuordnungen mit Kosten je Lkw und Abweichungskosten je Lkw und Tor (nach Wartezeit an Tor h, wenn nur der Lkw wechselt)."""

    def __init__(self, inst):
        if inst.m ** inst.n > C.ENUM_MAX_ASSIGNMENTS:
            raise ValueError("zu viele Zuordnungen")
        n, m = inst.n, inst.m
        N = m ** n
        idx = np.arange(N, dtype=np.int64)
        A = np.empty((N, n), dtype=np.int64)
        for i in range(n):
            A[:, i] = idx % m
            idx //= m
        L = np.zeros((N, m))
        K = np.zeros((N, m))
        for i in range(n):
            L[np.arange(N), A[:, i]] += inst.w[i]
            K[np.arange(N), A[:, i]] += 1.0
        wait = inst.a[None, :] + inst.b[None, :] * L
        self.inst, self.A = inst, A
        self.own = np.take_along_axis(wait, A, axis=1)                                  # (N, n) Wartezeit von Lkw i
        dev = np.empty((N, n, m))
        for i in range(n):
            Li = L.copy()
            Li[np.arange(N), A[:, i]] -= inst.w[i]
            dev[:, i, :] = inst.a[None, :] + inst.b[None, :] * (Li + inst.w[i])      # Wartezeit von Lkw i an Tor h bei unveränderten anderen
        self.dev = dev
        self.social = self.own.sum(axis=1)
        self.N = N


def _obedience(st):
    """Ungleichungen A sigma <= 0: je Lkw i und Paar (g, h), g != h: sum_{s: s_i = g} sigma(s) (own_i(s) - dev_i,h(s)) <= 0."""
    n, m, N = st.inst.n, st.inst.m, st.N
    rows, cols, vals = [], [], []
    r = 0
    for i in range(n):
        for g in range(m):
            sel = np.flatnonzero(st.A[:, i] == g)
            for h in range(m):
                if h == g:
                    continue
                rows.append(np.full(len(sel), r))
                cols.append(sel)
                vals.append(st.own[sel, i] - st.dev[sel, i, h])
                r += 1
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, N)).tocsr()


def _coarse(st):
    """Ungleichungen für CCE: je Lkw i und festes Tor h: sum_s sigma(s) (own_i(s) - dev_i,h(s)) <= 0."""
    n, m, N = st.inst.n, st.inst.m, st.N
    M = np.empty((n * m, N))
    r = 0
    for i in range(n):
        for h in range(m):
            M[r] = st.own[:, i] - st.dev[:, i, h]
            r += 1
    return M


def solve(st, kind="ce", sense="min"):
    """Beste (min) oder schlechteste (max) Summe der Wartezeiten unter allen korrelierten (kind='ce') bzw. groben korrelierten (kind='cce') Gleichgewichten.
    Rückgabe: (Zielwert, Verteilung sigma) oder (None, None), falls nicht lösbar."""
    A_ub = _obedience(st) if kind == "ce" else _coarse(st)
    c = st.social if sense == "min" else -st.social
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(A_ub.shape[0]), A_eq=np.ones((1, st.N)), b_eq=[1.0], bounds=(0, None), method="highs")
    if not res.success:
        return None, None
    sigma = np.maximum(res.x, 0.0)
    sigma /= sigma.sum()
    return float(st.social @ sigma), sigma


def ce_gap(st, sigma):
    """Größte Verbesserung, die sich ein Lkw durch Ersetzen einer Empfehlung g durch ein festes Tor h holen kann (Erwartungswert über die Verteilung, nicht bedingt), und die entsprechende
    Größe für feste Abweichungen ohne Kenntnis der Empfehlung (CCE-Lücke)."""
    n, m = st.inst.n, st.inst.m
    gap_ce = gap_cce = 0.0
    for i in range(n):
        for h in range(m):
            diff = st.own[:, i] - st.dev[:, i, h]
            gap_cce = max(gap_cce, float(sigma @ diff))
            for g in range(m):
                if g != h:
                    gap_ce = max(gap_ce, float(sigma[st.A[:, i] == g] @ diff[st.A[:, i] == g]))
    return gap_ce, gap_cce
