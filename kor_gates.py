"""Vehikel "Torwahl" aus nash-demo (Erzeugung, Kosten, Best-Response, Vollaufzählung wortgleich übernommen): Tor g hat die Wartezeit a_g + b_g * Last,
Lkw i hat Größe w_i; Kosten eines Lkw = Wartezeit seines Tors; Gütemaß = Summe der Wartezeiten aller Lkw."""

from dataclasses import dataclass

import numpy as np

import kor_constants as C


@dataclass(frozen=True)
class Instance:
    a: np.ndarray      # (m,) Grundwartezeit
    b: np.ndarray      # (m,) Zuschlag je Ladungseinheit
    w: np.ndarray      # (n,) Lkw-Größe
    seed: int

    @property
    def n(self):
        return len(self.w)

    @property
    def m(self):
        return len(self.a)


def generate(n, m, size_mode="mixed", seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(C.A_MIN, C.A_MAX, size=m)
    b = rng.uniform(C.B_MIN, C.B_MAX, size=m)
    if size_mode == "mixed":
        w = rng.choice(C.SIZES, size=n, p=C.SIZE_PROBS).astype(float)
    elif size_mode == "uniform":
        rng.choice(C.SIZES, size=n, p=C.SIZE_PROBS)      # Ziehung verbrauchen: gleiche Tore wie im gemischten Modus
        w = np.ones(n)
    else:
        raise ValueError(size_mode)
    return Instance(a, b, w, int(seed))


def loads(inst, assign):
    return np.bincount(assign, weights=inst.w, minlength=inst.m)


def waits(inst, assign):
    return inst.a + inst.b * loads(inst, assign)


def social_cost(inst, assign):
    """Summe der Wartezeiten aller Lkw."""
    return float(waits(inst, assign)[assign].sum())


def deviation_costs(inst, assign, i):
    """Wartezeit von Lkw i an jedem Tor, wenn nur er wechselt (an seinem eigenen Tor: die aktuelle Wartezeit)."""
    L = loads(inst, assign)
    L[assign[i]] -= inst.w[i]
    return inst.a + inst.b * (L + inst.w[i])


def best_response(inst, assign, i):
    """Bestes Tor bei unveränderten anderen; bei Gleichstand bleibt der Lkw."""
    costs = deviation_costs(inst, assign, i)
    cur = int(assign[i])
    best = int(np.argmin(costs))
    return cur if costs[cur] <= costs[best] + C.EPS else best


def is_equilibrium(inst, assign):
    return all(best_response(inst, assign, i) == assign[i] for i in range(inst.n))


def random_start(inst, seed):
    return np.random.default_rng(seed).integers(0, inst.m, size=inst.n)


def run_best_response(inst, start, seed=0):
    """Sequentielle Best-Response mit zufälliger Reihenfolge, bis ein ganzer Durchgang ohne Zug bleibt (terminiert: Potenzialspiel)."""
    rng = np.random.default_rng(seed)
    cur = np.array(start, dtype=np.int64)
    while True:
        moved = False
        for i in rng.permutation(inst.n):
            br = best_response(inst, cur, int(i))
            if br != cur[i]:
                cur[i] = br
                moved = True
        if not moved:
            return cur


def enumerable(inst):
    return inst.m ** inst.n <= C.ENUM_MAX_ASSIGNMENTS


def analyse(inst):
    """Vollaufzählung: alle reinen Gleichgewichte (direkt aus der Definition) und das Optimum (kleinste Summe der Wartezeiten)."""
    if not enumerable(inst):
        raise ValueError("zu viele Zuordnungen für die Vollaufzählung")
    N = inst.m ** inst.n
    idx = np.arange(N, dtype=np.int64)
    A = np.empty((N, inst.n), dtype=np.int8)
    for i in range(inst.n):
        A[:, i] = idx % inst.m
        idx //= inst.m
    L = np.zeros((N, inst.m))
    for i in range(inst.n):
        L[np.arange(N), A[:, i]] += inst.w[i]
    cost_gate = inst.a[None, :] + inst.b[None, :] * L
    truck_cost = np.take_along_axis(cost_gate, A.astype(np.int64), axis=1)
    social = truck_cost.sum(axis=1)
    is_ne = np.ones(N, dtype=bool)
    for i in range(inst.n):
        Li = L.copy()
        Li[np.arange(N), A[:, i]] -= inst.w[i]
        dev = inst.a[None, :] + inst.b[None, :] * (Li + inst.w[i])
        is_ne &= truck_cost[:, i] <= dev.min(axis=1) + C.EPS
    ne_idx = np.flatnonzero(is_ne)
    return {
        "n_ne": int(len(ne_idx)),
        "ne_assignments": A[ne_idx],
        "ne_social": social[ne_idx],
        "opt_cost": float(social.min()),
        "ne_cost_min": float(social[ne_idx].min()),
        "ne_cost_max": float(social[ne_idx].max()),
    }
