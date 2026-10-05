"""Lernen im wiederholten Torwahl-Spiel mit voller Information: Hedge (externer Regret, wie noregret-demo) und Regret-Matching (Hart/Mas-Colell 2000, bedingter/interner Regret).
Jeden Tag zieht jeder Lkw sein Tor und erfährt danach, was jedes Tor bei den gezogenen Toren der anderen gekostet hätte.

Regret-Matching: D_i(g, h) = Summe über alle bisherigen Tage mit gezogenem Tor g von (bezahlte Wartezeit - Wartezeit an Tor h). Hat der Lkw gestern g gezogen, wechselt er heute zu h != g mit
Wahrscheinlichkeit max(D_i(g, h), 0) / (t * mu) und bleibt sonst bei g. Kleine Wechselwahrscheinlichkeit = träge, aber die empirische Verteilung des Spiels nähert sich der Menge der
korrelierten Gleichgewichte."""

from dataclasses import dataclass

import numpy as np

import kor_constants as C


def normalizer(inst):
    return float((inst.a + inst.b * inst.w.sum()).max())


def cost_matrix(inst, g):
    """Wartezeit von Lkw i an Tor h bei den gezogenen Toren g der anderen."""
    L = np.bincount(g, weights=inst.w, minlength=inst.m)
    L_others = L[None, :] - np.where(np.arange(inst.m)[None, :] == g[:, None], inst.w[:, None], 0.0)
    return inst.a[None, :] + inst.b[None, :] * (inst.w[:, None] + L_others)


@dataclass
class Learning:
    method: str
    T: int
    assign: np.ndarray      # (T, n) gezogene Tore
    own: np.ndarray         # (T, n) bezahlte Wartezeit
    cf: np.ndarray          # (T, n, m) Wartezeit an jedem Tor bei den Toren der anderen
    switches: np.ndarray    # (T,) Zahl der Lkw, die heute ein anderes Tor ziehen als gestern
    Q: np.ndarray           # (T, n, m) Ziehungswahrscheinlichkeiten vor jedem Tag

    @property
    def social(self):
        return self.own.sum(axis=1)


def _softmax(S):
    """Gewichte proportional zu exp(S_h), stabil über das Maximum; Gewichte unter dem Gleitkomma-Minimum werden exakt 0 (keine künstliche Untergrenze)."""
    e = np.exp(S - S.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sample(rng, Q):
    return np.minimum((rng.random(len(Q))[:, None] > np.cumsum(Q, axis=1)).sum(axis=1), Q.shape[1] - 1)


def run_learning(inst, method="rm", T=3000, seed=0, eta=C.HEDGE_ETA, mu_factor=C.RM_MU_FACTOR):
    if method not in C.LEARN_METHODS:
        raise ValueError(method)
    rng = np.random.default_rng(seed)
    n, m = inst.n, inst.m
    cmax = normalizer(inst)
    rows = np.arange(n)
    mu = mu_factor * (m - 1) * cmax
    S = np.zeros((n, m))            # Hedge: Log-Gewichte (Summe von -eta * Verlust), exakt ohne Untergrenze
    P = _softmax(S)
    D = np.zeros((n, m, m))
    prev = None
    assign, own, cf = np.empty((T, n), dtype=np.int64), np.empty((T, n)), np.empty((T, n, m))
    switches = np.zeros(T, dtype=int)
    Q_hist = np.empty((T, n, m))
    for t in range(T):
        if method == "rm" and prev is not None:
            Q = np.maximum(D[rows, prev, :], 0.0) / (t * mu)
            Q[rows, prev] = 0.0
            Q[rows, prev] = 1.0 - Q.sum(axis=1)
        else:
            Q = P
        Q_hist[t] = Q
        g = _sample(rng, Q)
        cost = cost_matrix(inst, g)
        assign[t], own[t], cf[t] = g, cost[rows, g], cost
        if prev is not None:
            switches[t] = int((g != prev).sum())
        prev = g
        if method == "rm":
            D[rows, g, :] += own[t][:, None] - cost
        else:
            S = S - eta * cost / cmax
            P = _softmax(S)
    return Learning(method, T, assign, own, cf, switches, Q_hist)


def gap_curves(inst, lr):
    """Relativer CE-Gap und CCE-Gap der empirischen Verteilung nach jedem Tag.
    CE-Gap = größte Verbesserung, die sich ein Lkw durch Ersetzen seiner gezogenen Tor g durch ein festes Tor h holt (Summe über die Tage mit Ziehung g, geteilt durch die Zahl der Tage);
    CCE-Gap = derselbe Regret gegen ein festes Tor über alle Tage (externer Regret / Tage). Beide geteilt durch die mittlere bezahlte Wartezeit eines Lkw je Tag."""
    T, n, m = lr.T, inst.n, inst.m
    X = np.zeros((T, n, m, m))
    t_idx, i_idx = np.meshgrid(np.arange(T), np.arange(n), indexing="ij")
    X[t_idx, i_idx, lr.assign, :] = lr.own[:, :, None] - lr.cf                      # X[t, i, g, h] = 1[g_i^t = g] * (bezahlt - Wartezeit an h)
    R = np.cumsum(X, axis=0)                                                       # (T, n, m, m)
    paid = np.cumsum(lr.own, axis=0)                                                # (T, n)
    scale = paid.mean(axis=1)                                                       # mittlere bezahlte Gesamtwartezeit eines Lkw nach t Tagen
    ce = np.maximum(R, 0.0).reshape(T, -1).max(axis=1) / scale
    cce = (paid - np.cumsum(lr.cf, axis=0).min(axis=2)).max(axis=1) / scale
    return ce, cce
