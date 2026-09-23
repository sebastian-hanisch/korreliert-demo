# 🚦 Korrelierte Gleichgewichte – wenn ein Vermittler nur Empfehlungen ausspricht

Fünftes Stück der **Spieltheorie-&-Mechanism-Design-Linie** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Nachfolger von
[noregret-demo](https://sebastianhanisch-noregret-demo.streamlit.app/): dort hieß kleiner Regret "grobes korreliertes Gleichgewicht" – hier die stärkere Bedingung, bei der jeder Lkw seine
Empfehlung kennt. Der Vermittler ist ein Eingriff von außen ohne Zwang und ohne Preis; Eigennutz und volle Information über die Verteilung bleiben Modellannahmen.

## Warum dieses Problem

Im Nash-Gleichgewicht wählt jeder Lkw sein Tor unabhängig von den anderen. Ein **Vermittler** (eine Ampel, eine Disposition, ein Buchungssystem) kann mehr: Er zieht vor dem Start eine ganze Zuordnung aus
einer Verteilung, die nur er kennt, und teilt **jedem Lkw nur dessen Tor** mit. Der Lkw darf folgen oder nicht. Die Verteilung ist ein **korreliertes Gleichgewicht** (Aumann 1974), wenn Folgen für jeden Lkw und jede
mögliche Empfehlung mindestens so gut ist wie jedes andere feste Tor – gegeben, was die Empfehlung über die anderen verrät.

## Modell

**Torwahl-Vehikel** (`kor_gates.py`): wortgleich aus [nash-demo](https://github.com/sebastian-hanisch/nash-demo) (Tore mit Wartezeit $a_g + b_g \cdot$ Last, Lkw mit Größe 1, 2 oder 3; Gütemaß = Summe der Wartezeiten aller Lkw).

**Vermittler als LP** (`kor_ce.py`): Variable $\sigma(s) \ge 0$ für jede der $m^n$ Zuordnungen, $\sum_s \sigma(s) = 1$, und je Lkw $i$ und Tore $g \ne h$ die Gehorsamsbedingung
$\sum_{s: s_i = g} \sigma(s)\,[c_i(s) - c_i(h, s_{-i})] \le 0$. Zielfunktion: erwartete Summe der Wartezeiten, minimiert (bester Vermittler) oder maximiert (schlechtester gehorsamer). Das **grobe korrelierte
Gleichgewicht** verlangt nur $\sum_s \sigma(s)\,[c_i(s) - c_i(h, s_{-i})] \le 0$ je Lkw und festem Tor. Es gilt Nash (rein) ⊆ korreliert ⊆ grob korreliert. Gelöst mit `scipy.optimize.linprog` (HiGHS); wegen
$m^n$ Variablen sind höchstens 10 Lkw bei 3 Toren zugelassen.

**Lernen** (`kor_learning.py`): **Regret-Matching** (Hart/Mas-Colell 2000): der bedingte Regret $D_i(g,h)$ über alle Tage mit gezogenem Tor $g$ bestimmt, mit welcher Wahrscheinlichkeit ein Lkw, der gestern $g$ gezogen hat, heute zu $h$ wechselt.
Zum Vergleich **Hedge** (externer Regret, wie in noregret-demo). Gemessen wird die **CE-Lücke** der empirischen Verteilung des Spiels: $\max_{i,g,h} \frac1T D_i^T(g,h)$ geteilt durch die mittlere bezahlte Wartezeit.

**Mini-Spiel** (`kor_bimatrix.py`): zwei Lkw, zwei Tore (wie nash-demo), der Vermittler als kleines LP über die vier Zuordnungen, auch mit der Zielfunktion "gerecht" (kleinere größere erwartete Wartezeit).

## Methodik

- Alle LP-Werte sind eindeutige Optimalwerte fester Instanzen; das Optimum, die reinen Gleichgewichte und die Vermittler werden exakt aus der Definition gerechnet (kein Sampling).
- **Gehorsam** wird unabhängig vom LP nachgeprüft: über alle Zuordnungen direkt aus der Definition (Einzel-Lkw-Kosten), und die Lösung des LP gegen ein zweites Lösungsverfahren (`highs-ipm`).
- **Lernen** ist ein Zufallsprozess (Ziehungen der Tore) mit festen Seeds; die CE-Lücke wird zusätzlich gegen die LP-Auswertung derselben empirischen Verteilung (`kor_ce.ce_gap`) geprüft.
- **Literatur** (per Recherche geprüft, nicht nachgebaut): Aumann 1974 (korrelierte Gleichgewichte), Hart/Mas-Colell 2000 (Regret-Matching), Foster/Vohra 1997 (kalibriertes Lernen), Blum/Mansour 2007 (Swap-Regret).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Schlägt ein Vermittler das beste reine Gleichgewicht? | Standardinstanz (8 Lkw, 3 Tore, Seed 35): Optimum 95,5 min, reine Gleichgewichte 96,9 bis 100,3, bester Vermittler 96,1 – er schließt 54 % der Lücke zum Optimum mit 9 verschiedenen Zuordnungen. | `test_preset_numbers`, `test_standardfall_structure` |
| Und im Mittel? | 60 Instanzen (6 Lkw, 3 Tore, gemischte Größen): bester Vermittler im Mittel 1,5 % über dem Optimum, bestes reines Gleichgewicht 2,5 %, bestes grobes korreliertes 1,1 %. In 50 % der Instanzen schlägt der Vermittler jedes reine Gleichgewicht, in 35 % trifft er das Optimum; das grobe korrelierte schlägt das korrelierte in 38 %. Einheitliche Größen: 1,3 % / 2,1 % / 0,6 %. | `test_mediator_experiment_numbers` |
| Was kostet die größere Menge? | Das schlechteste korrelierte Gleichgewicht liegt im Mittel 19,0 % über dem Optimum (schlechtestes reines: 6,7 %), im Einzelfall bis 30 %; bei einheitlichen Größen 11,9 % gegen 2,1 %. Gehorsam allein garantiert keine gute Lösung. | dito |
| Wächst der Nutzen mit der Lkw-Zahl? | Ja: bei 4 Lkw schlägt der Vermittler das beste reine Gleichgewicht in 7 % der Instanzen, bei 8 Lkw in 73 %. Das beste reine Gleichgewicht liegt 0,7 % bzw. 2,9 % über dem Optimum, der beste Vermittler 0,6 % bzw. 1,8 %. Ob mehr Zuordnungen zum Mischen die Ursache sind, ist eine Vermutung, nicht gemessen. | `test_scaling_experiment_numbers` |
| Wann gewinnt er nichts? | Bei nur zwei Toren (8 Lkw, Seed 35) trifft schon ein reines Gleichgewicht das Optimum (217,3 min): der beste Vermittler ist gleich gut, der schlechteste (246,8) schlechter als jedes reine Gleichgewicht (224,0). | `test_zwei_tore_no_gain_and_kleine_instanz_cce_near_optimum` |
| Findet Lernen den Vermittler? | Regret-Matching und Hedge führen die CE-Lücke gegen null (nach 3000 Tagen 0,19 % bzw. 0,07 %, 20 Instanzen); Hedge ist schneller. Beide landen fast immer bei einem einzelnen reinen Gleichgewicht (letzter Tag ein Gleichgewicht: 95 % bzw. 100 %) und liegen 4,7 % bzw. 4,2 % über dem Optimum – nicht beim besten Vermittler (1,8 %; bestes reines Gleichgewicht 2,6 %). Lernen findet ein korreliertes Gleichgewicht, aber nicht das gute. | `test_learning_experiment_numbers` |
| Mini-Spiel | δ = 1: reine Gleichgewichte (A,B) mit Kosten (7, 8) und (B,A) mit (8, 7); gemischtes Gleichgewicht 75 % / 75 % mit erwarteten Kosten 8,5 für beide; die gerechte Ampel wechselt die beiden reinen Gleichgewichte ab: 7,5 für beide. Handrechnung, im Test nachgerechnet. | `test_mini_game_numbers_at_the_default_delta`, `test_fair_ampel_alternates_the_two_pure_equilibria_by_hand` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Vermittler kennt Tore, Größen und Kosten der Lkw** | Er löst ein LP mit einer Variable je Zuordnung ($m^n$) – ab etwa 10 Lkw nicht mehr komfortabel. Praktische Vermittler brauchen kleine Verteilungen. | Lernen (No-Regret, Regret-Matching) |
| **Alle folgen der Empfehlung** | Gehorsam ist eine Ungleichung für den Erwartungswert; wer Zusatzwissen hat, kann besser abweichen. | – |
| **Der Vermittler will das Beste für alle** | Ein Vermittler mit anderen Zielen findet ebenso gehorsame Verteilungen – das schlechteste korrelierte Gleichgewicht ist im Mittel deutlich schlechter als das schlechteste reine. | Mechanism Design mit Preisen ([Maut](https://sebastianhanisch-maut-demo.streamlit.app/)) |
| **Ein einmaliges Spiel** | Über viele Tage kann Lernen ein korreliertes Gleichgewicht erreichen, aber nicht das beste. | [No-Regret-Lernen](https://sebastianhanisch-noregret-demo.streamlit.app/) |
| **Alle entscheiden gleichzeitig, keiner legt sich fest** | Ein Anführer, der sich zuerst festlegt, kann das Ergebnis lenken – auch ohne Vermittler. | Stackelberg |

Nur **reine** Nash-Gleichgewichte werden aufgezählt; das gemischte Nash-Gleichgewicht kommt nur im Mini-Spiel vor. Die Messreihen gelten für diese eine Spielfamilie und diese Instanzgrößen (bis 8 Lkw in den Experimenten).

Verwandt: [noregret-demo](https://sebastianhanisch-noregret-demo.streamlit.app/) (Vorgänger), [nash-demo](https://sebastianhanisch-nash-demo.streamlit.app/) (Nash-Gleichgewichte im Torwahl-Spiel),
[maut-demo](https://sebastianhanisch-maut-demo.streamlit.app/) (Preise statt Empfehlungen).

## Tests

Pytest-Suite (`pytest tests/ -v`): Zustände, Kosten und Abweichungskosten gegen die Definition; Gehorsam der LP-Lösungen direkt aus der Definition; Inklusionen Nash ⊆ CE ⊆ CCE; jedes reine Gleichgewicht ist ein
korreliertes; LP-Werte gegen ein zweites Lösungsverfahren und das 2×2-LP gegen ein kleines LP im Mini-Spiel; Handrechnung im Mini-Spiel (Kostenmatrizen, gemischtes Gleichgewicht, gerechte Ampel);
Regret-Matching-Wahrscheinlichkeiten gegen eine unabhängige Schleife; CE- und CCE-Lücke gegen eine Schleife und gegen die LP-Auswertung der empirischen Verteilung; AppTest-Rauchtests (jedes Preset,
Empfehlungs-Check, Permalink-Grenzen, Mini-Spiel, drei Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README; LP-Werte exakt, Lernen mit großzügigen Bändern).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `kor_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Experiment-Seeds, Presets |
| `kor_presets.py` | Permalink/Presets-Mechanik |
| `kor_gates.py` | Torwahl-Vehikel, Vollaufzählung der reinen Gleichgewichte (aus nash-demo) |
| `kor_ce.py` | Zustände, LP für korrelierte und grobe korrelierte Gleichgewichte, CE-Lücke |
| `kor_bimatrix.py` | Mini-Spiel, Vermittler-LP über vier Zuordnungen |
| `kor_learning.py` | Regret-Matching, Hedge, Lücken der empirischen Verteilung |
| `kor_evaluation.py` | Analyse einer Instanz, drei Experimente |
| `kor_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Mehr als 10 Lkw (das LP wächst mit $m^n$) und mehr als 3 Tore.
- Sparsame oder strukturierte Vermittler (etwa Verteilungen mit wenigen Zuordnungen erzwingen).
- Die Frage, wer den Vermittler bezahlt und ob er die Empfehlung verfälschen kann (Mechanism Design mit Zahlungen).
- Ein PDF-Export – wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly, numpy und scipy.
