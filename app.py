"""Korrelierte Gleichgewichte - wenn ein Vermittler nur Empfehlungen ausspricht - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Linie "Spieltheorie & Mechanism Design" der "Konzepte"-Reihe (Nachfolger von noregret-demo): Ein Vermittler zieht eine Zuordnung aus einer Verteilung und teilt jedem Lkw nur sein
Tor mit. Wann ist das für alle folgsam - und was bringt es gegenüber Nash-Gleichgewichten?

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import kor_bimatrix as B
import kor_constants as C
from kor_evaluation import Settings, analyse, learning_experiment, mediator_experiment, scaling_experiment
from kor_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from kor_visualization import build_bimatrix, build_learning_gap, build_learning_outcome, build_mediator_experiment, build_range, build_recommendation, build_scaling

st.set_page_config(page_title="Korrelierte Gleichgewichte – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _mediator():
    return mediator_experiment("mixed"), mediator_experiment("uniform")


@st.cache_data(show_spinner=False)
def _scaling():
    return scaling_experiment()


@st.cache_data(show_spinner=False)
def _learning():
    return learning_experiment()


st.title("🚦 Korrelierte Gleichgewichte – wenn ein Vermittler nur Empfehlungen ausspricht")
st.markdown(
    """
Im Nash-Gleichgewicht wählt jeder Lkw sein Tor unabhängig von den anderen. Ein **Vermittler** (eine Ampel, eine Disposition, ein Buchungssystem) kann mehr: Er zieht vor dem Start eine ganze Zuordnung aus einer
Verteilung, die nur er kennt, und teilt **jedem Lkw nur dessen Tor** mit. Der Lkw darf dann folgen oder nicht. Die Empfehlung ist ein **korreliertes Gleichgewicht**, wenn folgen für jeden Lkw und jede
mögliche Empfehlung mindestens so gut ist wie jedes andere feste Tor - gegeben, was die Empfehlung über die anderen verrät. Die Demo berechnet den besten und den schlechtesten Vermittler exakt und fragt, wann er
sich lohnt, was er kostet und ob Lernen ihn findet.
"""
)
st.caption(
    "Fünftes Stück der Linie \"Spieltheorie & Mechanism Design\" der \"Konzepte\"-Reihe, Nachfolger von **noregret-demo**: dort hieß kleiner Regret \"grobes korreliertes Gleichgewicht\" - hier die "
    "stärkere Bedingung, bei der jeder Lkw seine Empfehlung kennt. Der Vermittler ist ein Eingriff von außen ohne Zwang und ohne Preis; Eigennutz und volle Information über die Verteilung bleiben Modellannahmen."
)

with st.expander("So funktioniert der Vermittler", expanded=True):
    st.markdown(
        """
1. **Verteilung.** Der Vermittler wählt eine Wahrscheinlichkeit $\\sigma(s)$ für jede der $m^n$ Zuordnungen $s$ und zieht daraus vor dem Start eine Zuordnung. Jeder Lkw erfährt **nur sein eigenes Tor**.
2. **Gehorsam.** Hat Lkw $i$ die Empfehlung $g$ bekommen, kennt er die Verteilung und kann daraus schließen, welche Zuordnungen der anderen dazu passen. Folgen ist nur dann sinnvoll, wenn seine
   erwartete Wartezeit dabei höchstens so groß ist wie bei jedem festen anderen Tor $h$. Das sind $n \\cdot m \\cdot (m-1)$ lineare Ungleichungen in $\\sigma$.
3. **Bester und schlechtester Vermittler.** Ein lineares Programm findet unter allen gehorsamen Verteilungen die mit der kleinsten (bester) bzw. größten (schlechtester) erwarteten Summe der Wartezeiten.
4. **Grob korreliert.** Verlangt man Gehorsam nur, bevor der Lkw seine Empfehlung kennt (ein festes Tor gegen die ganze Verteilung), erhält man das grobe korrelierte Gleichgewicht - eine größere Menge.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_trucks = st.slider("Lkw", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Lkw. Das LP hat eine Variable je Zuordnung (m^n): ab 10 Lkw dauert die Rechnung einige Sekunden.")
    m_gates = st.slider("Tore", *bounds("m_slider"), key="m_slider", step=C.M_STEP)
    size_mode = st.selectbox("Lkw-Größen", C.SIZE_MODES, key="size_select", format_func=lambda k: C.SIZE_MODE_LABELS[k],
                             help="Gemischt: Transporter, Lkw und Sattelzüge belegen ein Tor unterschiedlich stark. Einheitlich: alle Größe 1.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1, help="Legt Tore und Lkw-Größen fest.")
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Mini-Spiel**")
    delta = st.slider("δ – Tor B langsamer um [min]", *bounds("delta_slider"), key="delta_slider", step=C.MINI_DELTA_STEP, format="%.1f")

sync_query_params({"n_slider": int(n_trucks), "m_slider": int(m_gates), "size_select": size_mode, "delta_slider": float(delta), "seed_input": int(seed)})

settings = Settings(int(n_trucks), int(m_gates), size_mode, int(seed))
with st.spinner("Löse das lineare Programm..."):
    a = analyse(settings)
inst, en = a.inst, a.enum
r = a.ratios()

# --- Der Vermittler --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der Vermittler")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Optimum", f"{de(a.opt)} min", help="Kleinste Summe der Wartezeiten über alle Zuordnungen, von außen gesteuert und ohne Rücksicht auf die Anreize.")
m2.metric("Bestes reines Nash-Gleichgewicht", f"{de(en['ne_cost_min'])} min", delta=f"{pct(r['ne_min'] - 1, 1)} über dem Optimum", delta_color="off", help=f"Unter {en['n_ne']} reinen Gleichgewichten (Vollaufzählung).")
m3.metric("Bester Vermittler (korreliert)", f"{de(a.ce_min[0])} min", delta=f"{pct(r['ce_min'] - 1, 1)} über dem Optimum", delta_color="off", help=f"Gehorsame Verteilung mit {a.support_size('ce_min')} verschiedenen Zuordnungen.")
m4.metric("Bester grob korrelierter", f"{de(a.cce_min[0])} min", delta=f"{pct(r['cce_min'] - 1, 1)} über dem Optimum", delta_color="off", help="Nur Gehorsam vor Kenntnis der Empfehlung: die größere Menge.")

st.plotly_chart(build_range(r), width="stretch", key="range_chart")
st.caption("Jede Linie reicht vom besten bis zum schlechtesten Gleichgewicht der jeweiligen Sorte (Summe der Wartezeiten geteilt durch das Optimum). Reine Gleichgewichte sind korreliert (Verteilung auf eine Zuordnung), "
           "korrelierte sind grob korreliert - die Mengen wachsen, deshalb wird das beste Ergebnis besser, das schlechteste schlechter.")

gap_closed = (en["ne_cost_min"] - a.ce_min[0]) / max(en["ne_cost_min"] - a.opt, 1e-12)
if a.ce_min[0] < en["ne_cost_min"] - 1e-9:
    st.success(f"✅ Der beste Vermittler schlägt jedes reine Gleichgewicht: {de(a.ce_min[0])} statt {de(en['ne_cost_min'])} min - er schließt {pct(gap_closed)} der Lücke zum Optimum ({de(a.opt)} min). "
               f"Dafür braucht er {a.support_size('ce_min')} verschiedene Zuordnungen. Der schlechteste gehorsame Vermittler liegt bei {de(a.ce_max[0])} min, schlechter als das schlechteste reine Gleichgewicht ({de(en['ne_cost_max'])} min).")
else:
    st.info(f"Hier gewinnt der Vermittler nichts: das beste korrelierte Gleichgewicht ({de(a.ce_min[0])} min) ist bereits ein reines Nash-Gleichgewicht"
            f"{' und trifft das Optimum' if abs(a.ce_min[0] - a.opt) < 1e-9 else ''}. Der schlechteste gehorsame Vermittler liegt bei {de(a.ce_max[0])} min - {de(a.ce_max[0] - en['ne_cost_max'])} min über dem schlechtesten reinen Gleichgewicht.")

st.markdown("**Was der beste Vermittler empfiehlt** (die wahrscheinlichsten Zuordnungen; Zahl = Tor je Lkw)")
top = a.top_assignments("ce_min", 8)
st.dataframe({"Wahrscheinlichkeit": [pct(p, 1) for _, p, _ in top], **{f"Lkw {i + 1}": [f"Tor {s[i] + 1}" for s, _, _ in top] for i in range(inst.n)}, "Summe der Wartezeiten (min)": [de(c, 2) for _, _, c in top]}, hide_index=True)
st.caption(f"Insgesamt {a.support_size('ce_min')} Zuordnungen mit positiver Wahrscheinlichkeit; gezeigt sind die {len(top)} wahrscheinlichsten. Der Vermittler mischt Zuordnungen, die einzeln gar nicht gehorsam wären "
           "(etwa ein Lkw an einem überfüllten Tor) - gehorsam ist erst die Verteilung.")

st.markdown("**Empfehlungs-Check: warum folgt der Lkw?**")
for _key, _hi in (("check_truck", inst.n), ("check_gate", inst.m)):
    if _key in st.session_state:
        st.session_state[_key] = min(int(st.session_state[_key]), _hi)     # Regler-Grenzen hängen von Lkw- und Torzahl ab
c1, c2, c3 = st.columns(3)
kind = c1.radio("Vermittler", ("ce_min", "ce_max"), format_func=lambda k: "Bester" if k == "ce_min" else "Schlechtester", horizontal=True, key="check_kind")
truck = c2.slider("Lkw", 1, inst.n, key="check_truck")
gate = c3.slider("Empfohlenes Tor", 1, inst.m, key="check_gate")
vals, prob = a.follow_or_deviate(kind, truck - 1, gate - 1)
if vals is None:
    st.info(f"Diese Empfehlung (Lkw {truck} → Tor {gate}) spricht der {'beste' if kind == 'ce_min' else 'schlechteste'} Vermittler nie aus.")
else:
    st.plotly_chart(build_recommendation(vals, gate - 1, truck), width="stretch", key=f"rec_chart_{kind}_{truck}_{gate}")
    st.caption(f"Lkw {truck} bekommt Tor {gate} in {pct(prob, 1)} der Fälle empfohlen. Unter dieser Bedingung wartet er bei Gehorsam im Mittel {de(vals[gate - 1], 2)} min; jedes andere feste Tor wäre "
               f"{'nicht besser' if vals[gate - 1] <= vals.min() + 1e-6 else 'besser'} (kleinster Wert {de(vals.min(), 2)} min). Genau das verlangt die Bedingung für alle Lkw und alle Empfehlungen.")

st.markdown("---")

# --- Experiment 1: wie viel bringt der Vermittler? -----------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viel bringt der Vermittler - und was kostet er?")
st.caption(f"{C.MEDIATOR_N} Lkw, {C.MEDIATOR_M} Tore, {len(C.MEDIATOR_SEEDS)} feste Instanzen je Größenmodus; beste und schlechteste Werte aller reinen, korrelierten und grob korrelierten Gleichgewichte, "
           "jeweils gemittelt und geteilt durch das Optimum der Instanz.")
if st.button("Vermittler berechnen (dauert etwa 5 Sekunden)", key="mediator_start"):
    st.session_state["mediator_on"] = True
if st.session_state.get("mediator_on"):
    with st.spinner("Rechne 480 lineare Programme..."):
        res_m, res_u = _mediator()
    st.plotly_chart(build_mediator_experiment(res_m, res_u), width="stretch", key="mediator_chart")
    st.warning(
        f"**Befund (gemischte Größen):** Im Mittel liegt der beste Vermittler {pct(res_m['ce_min_mean'] - 1, 1)} über dem Optimum, das beste reine Nash-Gleichgewicht {pct(res_m['ne_min_mean'] - 1, 1)}, "
        f"das beste grobe korrelierte Gleichgewicht {pct(res_m['cce_min_mean'] - 1, 1)}. In {pct(res_m['share_ce_beats_ne'])} der Instanzen schlägt der Vermittler jedes reine Gleichgewicht, in "
        f"{pct(res_m['share_ce_optimal'])} trifft er das Optimum. Der Preis: das schlechteste korrelierte Gleichgewicht liegt im Mittel {pct(res_m['ce_max_mean'] - 1, 1)} über dem Optimum (bei den reinen nur {pct(res_m['ne_max_mean'] - 1, 1)}), "
        f"im Einzelfall bis {pct(res_m['ce_max_max'] - 1)}. Mit einheitlichen Größen ist das Bild ähnlich (bester Vermittler {pct(res_u['ce_min_mean'] - 1, 1)}, bestes reines {pct(res_u['ne_min_mean'] - 1, 1)}, "
        f"bestes grobes {pct(res_u['cce_min_mean'] - 1, 1)}). Ein Vermittler ist nur so gut wie seine Verteilung: Gehorsam allein garantiert keine gute Lösung."
    )

st.markdown("---")

# --- Experiment 2: Skalierung ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wächst der Nutzen mit der Lkw-Zahl?")
st.caption(f"{len(C.SCALING_SEEDS)} feste Instanzen je Lkw-Zahl von {C.SCALING_NS[0]} bis {C.SCALING_NS[-1]}, {C.MEDIATOR_M} Tore, gemischte Größen; Mittel der jeweils besten Werte.")
if st.button("Skalierung rechnen (dauert etwa 15 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne..."):
        rows_s = _scaling()
    st.plotly_chart(build_scaling(rows_s), width="stretch", key="scaling_chart")
    lo, hi = rows_s[0], rows_s[-1]
    st.warning(
        f"**Befund:** Bei {lo['n']} Lkw schlägt der Vermittler das beste reine Gleichgewicht nur in {pct(lo['share_ce_beats_ne'])} der Instanzen, bei {hi['n']} Lkw in {pct(hi['share_ce_beats_ne'])}. "
        f"Der Abstand des besten reinen Gleichgewichts zum Optimum liegt bei {pct(lo['ne_min'] - 1, 1)} bzw. {pct(hi['ne_min'] - 1, 1)}, der des besten Vermittlers bei {pct(lo['ce_min'] - 1, 1)} bzw. {pct(hi['ce_min'] - 1, 1)}: "
        "der Vermittler schließt bei mehr Lkw einen größeren Teil der Lücke - vermutlich, weil er mehr Zuordnungen zum Mischen hat."
    )

st.markdown("---")

# --- Experiment 3: Lernen -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Findet Lernen den Vermittler?")
st.caption(f"{len(C.LEARN_EXP_SEEDS)} feste Instanzen ({C.LEARN_EXP_N} Lkw, {C.LEARN_EXP_M} Tore, gemischte Größen), {C.LEARN_EXP_T} Tage. Regret-Matching (Hart/Mas-Colell 2000: Wechselwahrscheinlichkeit proportional zum bedingten Regret) gegen Hedge "
           f"(η = {C.HEDGE_ETA:g}, wie in noregret-demo). Gemessen wird die **CE-Lücke** der empirischen Verteilung des Spiels: die größte Verbesserung, die sich ein Lkw durch Ersetzen einer gezogenen Empfehlung durch "
           "ein festes Tor holen kann, geteilt durch die bezahlte Wartezeit.")
if st.button("Lernen vergleichen (dauert etwa 15 Sekunden)", key="learning_start"):
    st.session_state["learning_on"] = True
if st.session_state.get("learning_on"):
    with st.spinner("Lerne..."):
        res_l = _learning()
    l1, l2 = st.columns(2)
    l1.plotly_chart(build_learning_gap(res_l), width="stretch", key="learning_gap_chart")
    l2.plotly_chart(build_learning_outcome(res_l), width="stretch", key="learning_outcome_chart")
    rm, hd = res_l["rm"], res_l["hedge"]
    st.warning(
        f"**Befund:** Beide Verfahren führen die CE-Lücke gegen null: nach {res_l['days'][-1]} Tagen {pct(rm['ce_gap'][-1], 2)} bei Regret-Matching und {pct(hd['ce_gap'][-1], 2)} bei Hedge (Mittel über {res_l['n_inst']} Instanzen). "
        f"Hedge ist hier schneller - der externe Regret genügt in diesem Spiel praktisch auch für das korrelierte Gleichgewicht. Aber beide landen fast immer bei einem einzelnen reinen Gleichgewicht "
        f"(letzter Tag ein Nash-Gleichgewicht: {pct(rm['last_is_ne'])} bzw. {pct(hd['last_is_ne'])}) und damit {pct(rm['window'] - 1, 1)} bzw. {pct(hd['window'] - 1, 1)} über dem Optimum - nicht beim besten Vermittler "
        f"({pct(res_l['ref_ce_min'] - 1, 1)} über dem Optimum; bestes reines Gleichgewicht {pct(res_l['ref_ne_min'] - 1, 1)}). Lernen findet ein korreliertes Gleichgewicht, aber nicht das gute: "
        "welches, entscheidet nicht die Gehorsamsbedingung, sondern das Verfahren."
    )

st.markdown("---")

# --- Mini-Spiel --------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚦 Mini-Spiel: zwei Lkw, zwei Tore - die Ampel")
st.caption(f"Beide Lkw Größe 1, beide Tore Grundzeit {C.MINI_A:g} min und Zuschlag {C.MINI_B:g} min; Tor B ist um δ Minuten langsamer (Regler links). Kosten = Wartezeit (kleiner ist besser).")
K1, K2 = B.cost_matrices(float(delta))
names = ("Tor A", "Tor B")
table = "| Lkw 1 ↓ / Lkw 2 → | Tor A | Tor B |\n|---|---|---|\n" + "\n".join(f"| **{names[i]}** | " + " | ".join(f"{K1[i, j]:g} / {K2[i, j]:g}" for j in range(2)) + " |" for i in range(2))
st.markdown(table)
pure = B.pure_equilibria(K1, K2)
mixed = B.mixed_equilibrium(K1, K2)
fair = B.mediator(K1, K2, "fair")
fair_costs = B.expected_costs(K1, K2, fair)
chart = {}
if pure:
    g1, g2 = pure[0]
    chart[f"Reines Gleichgewicht ({names[g1]}, {names[g2]})"] = (float(K1[g1, g2]), float(K2[g1, g2]))
if mixed is not None:
    p, q = mixed
    sigma = np.outer([p, 1 - p], [q, 1 - q])
    chart["Gemischtes Gleichgewicht"] = B.expected_costs(K1, K2, sigma)
chart["Gerechte Ampel"] = fair_costs
st.markdown("**Reine Gleichgewichte:** " + ("; ".join(f"(Lkw 1 → {names[a1]}, Lkw 2 → {names[a2]})" for a1, a2 in pure) or "keines"))
if mixed is not None:
    st.markdown(f"**Gemischtes Gleichgewicht:** Lkw 1 wählt Tor A mit Wahrscheinlichkeit {pct(mixed[0])}, Lkw 2 mit {pct(mixed[1])} - beide unabhängig voneinander.")
st.markdown("**Gerechte Ampel** (Vermittler, der die größere erwartete Wartezeit der beiden Lkw minimiert): " + ", ".join(f"{pct(fair[k])} auf (Lkw 1 → {names[B.STATES[k][0]]}, Lkw 2 → {names[B.STATES[k][1]]})" for k in range(4) if fair[k] > 1e-9))
st.plotly_chart(build_bimatrix(chart), width="stretch", key="bimatrix_chart")
if len(pure) >= 2 and mixed is not None:
    st.caption("Zwei Lkw, die sich aus dem Weg gehen wollen, haben zwei reine Gleichgewichte - in jedem wartet ein Lkw länger - und ein gemischtes, in dem beide zu oft am selben Tor landen. Die Ampel wechselt die beiden "
               "reinen Gleichgewichte ab: jeder wartet im Mittel weniger als im gemischten Gleichgewicht, und beide gleich lang. Sie kann nichts erzwingen, aber folgen lohnt für beide.")
else:
    st.caption("Ab δ > 2 ist Tor A für beide die bessere Wahl, egal was der andere tut: es bleibt ein einziges Gleichgewicht und die Ampel hat nichts zu koordinieren. Bei δ = 2 genau ist Tor A nur noch gleich gut, wenn der andere Tor A nimmt (drei reine Gleichgewichte, kein gemischtes).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Vermittler kennt Tore, Größen und Kosten der Lkw** | Er löst ein LP mit einer Variable je Zuordnung ($m^n$) - schon ab etwa 10 Lkw ist das nicht mehr komfortabel. Praktische Vermittler brauchen kleine Verteilungen. | Lernen (No-Regret, Regret-Matching) |
| **Alle folgen der Empfehlung** | Gehorsam ist nur eine Ungleichung für den Erwartungswert; wer Zusatzwissen hat (etwa Wartezeiten aus anderen Quellen), kann besser abweichen. | - |
| **Der Vermittler will das Beste für alle** | Ein Vermittler mit anderen Zielen findet ebenso gehorsame Verteilungen - das schlechteste korrelierte Gleichgewicht ist im Mittel deutlich schlechter als das schlechteste reine (Experiment oben). | Mechanism Design mit Preisen ([Maut](https://sebastianhanisch-maut-demo.streamlit.app/)) |
| **Ein einmaliges Spiel** | Über viele Tage kann Lernen ein korreliertes Gleichgewicht erreichen, aber nicht das beste (Experiment oben). | [No-Regret-Lernen](https://sebastianhanisch-noregret-demo.streamlit.app/) |
| **Alle entscheiden gleichzeitig, keiner legt sich fest** | Ein Anführer, der sich zuerst festlegt, kann das Ergebnis lenken - auch ohne Vermittler. | **Stackelberg** |
"""
)
st.caption(
    "Verwandt: [noregret-demo](https://sebastianhanisch-noregret-demo.streamlit.app/) (Vorgänger: grobes korreliertes Gleichgewicht durch Regret-Lernen), "
    "[nash-demo](https://sebastianhanisch-nash-demo.streamlit.app/) (Nash-Gleichgewichte im Torwahl-Spiel), [maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise statt Empfehlungen)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kosten.** Lkw $i$ mit Größe $w_i$, Tore $g$ mit $a_g, b_g$: Wartezeit $c_i(s) = a_{s_i} + b_{s_i} \sum_{j: s_j = s_i} w_j$; Summe der Wartezeiten $SC(s) = \sum_i c_i(s)$.

**Korreliertes Gleichgewicht** (Aumann 1974). Verteilung $\sigma$ über Zuordnungen mit $\sum_{s: s_i = g} \sigma(s)\,\big[c_i(s) - c_i(h, s_{-i})\big] \le 0$ für alle Lkw $i$ und alle Tore $g \ne h$.
Der Vermittler zieht $s \sim \sigma$ und teilt jedem Lkw nur $s_i$ mit. Es sind $n\,m\,(m-1)$ lineare Ungleichungen; unter den Verteilungen minimiert bzw. maximiert ein LP $\sum_s \sigma(s)\,SC(s)$.

**Grobes korreliertes Gleichgewicht.** $\sum_s \sigma(s)\,\big[c_i(s) - c_i(h, s_{-i})\big] \le 0$ für alle $i, h$ - eine Bedingung je Lkw und Tor, ohne bedingt auf die Empfehlung zu sein. Es gilt
Nash (rein) $\subseteq$ CE $\subseteq$ CCE.

**Regret-Matching** (Hart/Mas-Colell 2000). $D_i^t(g,h) = \sum_{\tau \le t,\, s_i^\tau = g} \big[c_i(s^\tau) - c_i(h, s_{-i}^\tau)\big]$. Hat $i$ am Tag $t$ das Tor $g$ gezogen, wählt er am Tag $t+1$ das Tor $h \ne g$ mit
Wahrscheinlichkeit $\max(D_i^t(g,h), 0) / (t\,\mu)$ und sonst wieder $g$. Die empirische Verteilung der Zuordnungen konvergiert gegen die Menge der korrelierten Gleichgewichte.

**CE-Lücke** der empirischen Verteilung: $\max_{i,g,h} \frac1T D_i^T(g,h)$ (nicht negativ gemacht), geteilt durch die mittlere bezahlte Wartezeit.

Implementiert in `kor_ce.py` (LP mit `scipy.optimize.linprog`), `kor_learning.py` (Regret-Matching, Hedge, Lücken), `kor_bimatrix.py` (Mini-Spiel), `kor_evaluation.py` (Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spieltheorie: von Nash bis Myerson-Satterthwaite](https://sebastianhanisch.net/konzepte-spieltheorie.html)."
)
