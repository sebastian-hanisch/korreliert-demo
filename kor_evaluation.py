"""Auswertung: ein Vermittler-LP je Instanz und drei Experimente (Nutzen des Vermittlers, Skalierung, Lernen gegen Vermittler)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import kor_ce as CE
import kor_constants as C
import kor_gates as G
import kor_learning as L


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    size_mode: str = "mixed"
    seed: int = C.DEFAULT_SEED


@lru_cache(maxsize=256)
def instance(n, m, size_mode, seed):
    return G.generate(n, m, size_mode, seed)


KEYS = ("opt", "ne_min", "ce_min", "cce_min", "ne_max", "ce_max", "cce_max")


@dataclass
class Analysis:
    settings: Settings
    inst: object
    states: object
    enum: dict            # Optimum und reine Gleichgewichte aus der Vollaufzählung
    ce_min: tuple         # (Wert, Verteilung) bestes korreliertes Gleichgewicht
    ce_max: tuple
    cce_min: tuple
    cce_max: tuple

    @property
    def opt(self):
        return self.enum["opt_cost"]

    def ratios(self):
        """Summe der Wartezeiten geteilt durch das Optimum für die sieben Größen."""
        o = self.opt
        return {"opt": 1.0, "ne_min": self.enum["ne_cost_min"] / o, "ce_min": self.ce_min[0] / o, "cce_min": self.cce_min[0] / o,
                "ne_max": self.enum["ne_cost_max"] / o, "ce_max": self.ce_max[0] / o, "cce_max": self.cce_max[0] / o}

    def top_assignments(self, kind="ce_min", k=8):
        """Die k wahrscheinlichsten Zuordnungen des Vermittlers: (Zuordnung, Wahrscheinlichkeit, Summe der Wartezeiten)."""
        sigma = getattr(self, kind)[1]
        order = np.argsort(-sigma)[:k]
        return [(tuple(int(x) for x in self.states.A[s]), float(sigma[s]), float(self.states.social[s])) for s in order if sigma[s] > 1e-9]

    def support_size(self, kind="ce_min"):
        return int((getattr(self, kind)[1] > 1e-9).sum())

    def follow_or_deviate(self, kind, i, g):
        """Erwartete Wartezeit von Lkw i, dem der Vermittler Tor g empfohlen hat, wenn er (gegeben die Empfehlung) jedes feste Tor h wählt, und die Wahrscheinlichkeit der Empfehlung g.
        Bei h = g ist das die Wartezeit bei Gehorsam. Ohne diese Empfehlung: (None, 0)."""
        sigma = getattr(self, kind)[1]
        sel = self.states.A[:, i] == g
        p = float(sigma[sel].sum())
        if p < 1e-12:
            return None, 0.0
        return (sigma[sel][:, None] * self.states.dev[sel, i, :]).sum(axis=0) / p, p


@lru_cache(maxsize=64)
def analyse(settings):
    inst = instance(settings.n, settings.m, settings.size_mode, settings.seed)
    st = CE.States(inst)
    return Analysis(settings, inst, st, G.analyse(inst), CE.solve(st, "ce", "min"), CE.solve(st, "ce", "max"), CE.solve(st, "cce", "min"), CE.solve(st, "cce", "max"))


def _ratios_of(inst):
    st = CE.States(inst)
    en = G.analyse(inst)
    o = en["opt_cost"]
    return {"opt": 1.0, "ne_min": en["ne_cost_min"] / o, "ce_min": CE.solve(st, "ce", "min")[0] / o, "cce_min": CE.solve(st, "cce", "min")[0] / o,
            "ne_max": en["ne_cost_max"] / o, "ce_max": CE.solve(st, "ce", "max")[0] / o, "cce_max": CE.solve(st, "cce", "max")[0] / o}


def mediator_experiment(size_mode="mixed", n=None, m=None, seeds=None):
    """Über feste Instanzen: Mittel von sieben Größen (Summe der Wartezeiten / Optimum) und Anteile, in denen der Vermittler das beste Nash-Gleichgewicht schlägt bzw. das Optimum trifft."""
    n = C.MEDIATOR_N if n is None else n
    m = C.MEDIATOR_M if m is None else m
    seeds = C.MEDIATOR_SEEDS if seeds is None else seeds
    rows = [_ratios_of(instance(n, m, size_mode, s)) for s in seeds]
    arr = {k: np.array([r[k] for r in rows]) for k in KEYS}
    out = {f"{k}_mean": float(v.mean()) for k, v in arr.items()}
    out.update({f"{k}_max": float(v.max()) for k, v in arr.items()})
    out["share_ce_beats_ne"] = float(np.mean(arr["ce_min"] < arr["ne_min"] - 1e-9))
    out["share_cce_beats_ce"] = float(np.mean(arr["cce_min"] < arr["ce_min"] - 1e-9))
    out["share_ce_optimal"] = float(np.mean(arr["ce_min"] < 1 + 1e-9))
    out["share_cce_optimal"] = float(np.mean(arr["cce_min"] < 1 + 1e-9))
    out["n_inst"] = len(seeds)
    out["arrays"] = arr
    return out


def scaling_experiment(ns=None, size_mode="mixed", m=None, seeds=None):
    ns = C.SCALING_NS if ns is None else ns
    m = C.MEDIATOR_M if m is None else m
    seeds = C.SCALING_SEEDS if seeds is None else seeds
    rows = []
    for n in ns:
        r = mediator_experiment(size_mode, n, m, seeds)
        rows.append({"n": n, **{k: r[f"{k}_mean"] for k in KEYS}, "share_ce_beats_ne": r["share_ce_beats_ne"]})
    return rows


def learning_experiment(n=None, m=None, T=None, seeds=None, days=None, size_mode="mixed"):
    """Regret-Matching gegen Hedge: mittlerer relativer CE-Gap und CCE-Gap an ausgewählten Tagen; Ausgang der letzten 100 Tage gegen die LP-Werte derselben Instanzen."""
    n = C.LEARN_EXP_N if n is None else n
    m = C.LEARN_EXP_M if m is None else m
    T = C.LEARN_EXP_T if T is None else T
    seeds = C.LEARN_EXP_SEEDS if seeds is None else seeds
    days = tuple(d for d in C.CHECK_DAYS if d <= T) if days is None else days
    insts = [instance(n, m, size_mode, s) for s in seeds]
    refs = [_ratios_of(i) for i in insts]
    opts = [G.analyse(i)["opt_cost"] for i in insts]
    out = {"days": days, "n_inst": len(seeds), "ref_ne_min": float(np.mean([r["ne_min"] for r in refs])), "ref_ce_min": float(np.mean([r["ce_min"] for r in refs])),
           "ref_ne_max": float(np.mean([r["ne_max"] for r in refs])), "ref_ce_max": float(np.mean([r["ce_max"] for r in refs]))}
    for method in C.LEARN_METHODS:
        ce, cce, window, mean_all, pure = [], [], [], [], []
        for k, (inst, opt) in enumerate(zip(insts, opts)):
            lr = L.run_learning(inst, method, T, k)
            g_ce, g_cce = L.gap_curves(inst, lr)
            ce.append([g_ce[d - 1] for d in days])
            cce.append([g_cce[d - 1] for d in days])
            window.append(lr.social[-100:].mean() / opt)
            mean_all.append(lr.social.mean() / opt)
            pure.append(G.is_equilibrium(inst, lr.assign[-1]))
        out[method] = {"ce_gap": np.mean(ce, axis=0), "cce_gap": np.mean(cce, axis=0), "window": float(np.mean(window)), "mean_all": float(np.mean(mean_all)),
                       "last_is_ne": float(np.mean(pure)), "ce_gap_max_final": float(np.max(np.array(ce)[:, -1]))}
    return out
