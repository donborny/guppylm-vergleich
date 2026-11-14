# Anleitung: Ihr Gruppenergebnis im Plenum vorstellen

**Transformer, Sitzung 2 · Plenum (40 Minuten)**

Jede Gruppe hat ein Stück des Weges untersucht, den eine Frage durch Guppy nimmt. Im Plenum setzen Sie diese Stücke zu einem Ganzen zusammen. Jede Gruppe hat dafür **5 Minuten Vortrag und 1 Minute Rückfragen**.

## Reihenfolge: dem Weg der Daten folgen

Die Gruppen tragen nicht alphabetisch vor, sondern in der Reihenfolge, in der die Daten durch das Modell laufen:

| Nr. | Gruppe | Station | Übergabe an die nächste Gruppe |
|---|---|---|---|
| 1 | **A** · Trainingsbatch | Aus einem Dialog wird ein Batch mit `x`, `y` und Maske. | „Jetzt wissen wir, was ins Modell geht. Gruppe C zeigt, was drinnen damit passiert.“ |
| 2 | **C** · Was kann das Netz? | Embeddings und Positionen; was das fertige Netz kann. | „Die Vektoren laufen durch sechs Blöcke. Gruppe D zeigt, warum das funktioniert.“ |
| 3 | **D** · Residualverbindung | Warum das Lernsignal durch sechs Blöcke kommt. | „Das Signal kommt an. Aber welches Signal? Gruppe B zeigt, woher es kommt.“ |
| 4 | **B** · Loss und Backprop | Wie der Verlust die Gewichte verändert. | „Das Modell ist trainiert. Gruppe E zeigt, wie es damit antwortet.“ |
| 5 | **E** · Nächstes Token | Wie aus Logits Token für Token eine Antwort wird. | Abschluss, danach der gemeinsame Durchgang |

## So bauen Sie Ihre 5 Minuten auf

| Zeit | Teil | Inhalt |
|---|---|---|
| 0:30 | **Leitfrage** | Lesen Sie die Leitfrage Ihres Blatts vor. Ein Satz dazu, warum sie wichtig ist. |
| 2:00 | **Grafik** | Erklären Sie Ihre ergänzte Grafik von links nach rechts. Zeigen Sie dabei auf die Stelle, über die Sie gerade sprechen. |
| 1:30 | **Beleg** | Zeigen Sie Ihre Messung: Was haben Sie **erwartet**, was haben Sie **gemessen**? Eine nicht bestätigte Erwartung ist ein gutes Ergebnis, sagen Sie offen, was Sie überrascht hat. |
| 0:30 | **Kernaussage** | Ein bis zwei Sätze, die man sich merken soll. Schreiben Sie sie groß auf. |
| 0:30 | **Übergabe** | Ein Satz zur nächsten Station (siehe Tabelle oben). |

**Danach:** 1 Minute Rückfragen aus dem Plenum.

## Vorlage für Ihr Plakat oder Ihre Folie

```
┌─────────────────────────────────────────────────────────┐
│  Gruppe X: <Leitfrage>                                  │
├──────────────────────────────┬──────────────────────────┤
│                              │  Erwartung → Messung     │
│   Ihre ergänzte Grafik       │  ......... → .........   │
│   (vom Arbeitsblatt,         │  ......... → .........   │
│    mit Ihren Werten)         │                          │
│                              │  Überraschend war:       │
│                              │  ......................  │
├──────────────────────────────┴──────────────────────────┤
│  KERNAUSSAGE: ......................................... │
└─────────────────────────────────────────────────────────┘
```

## Rollen in der Gruppe

Teilen Sie die Rollen auf, damit alle vier etwas beitragen:

| Rolle | Aufgabe |
|---|---|
| **Moderation** | Leitfrage, Übergabe, achtet auf die Zeit |
| **Grafik** | erklärt die ergänzte Grafik |
| **Beleg** | stellt Erwartung und Messung vor |
| **Kernaussage** | formuliert die Kernaussage und beantwortet Rückfragen |

## Vorbereitung: die letzten 15 Minuten der Gruppenphase

- [ ] Grafik vom Arbeitsblatt abgezeichnet und mit **Ihren** Werten ergänzt
- [ ] Erwartung und Messung nebeneinander notiert
- [ ] Kernaussage in höchstens zwei Sätzen, groß geschrieben
- [ ] Übergabesatz zur nächsten Gruppe vorbereitet
- [ ] Rollen verteilt, einmal mit Stoppuhr geprobt (5 Minuten)

## Optional: Live-Vorführung im Browser

Die Gruppen **C** und **E** können ihre Aussage live zeigen, im [Checkpoint-Vergleich](https://donborny.github.io/guppylm-vergleich/). Er läuft ohne Installation im Browser.

- **Gruppe C:** Stellen Sie die bekannte, die umformulierte und die themenfremde Frage bei **Temperatur 0**.
- **Gruppe E:** Stellen Sie dieselbe Frage mit festem Seed bei Temperatur 0,1 und 1,5 und mit mehreren **Durchläufen**.
- Eigene trainierte Modelle (`.pt` aus dem Notebook) laden Sie per Drag & Drop ins Feld „Eigenen Transformer hochladen“. Die Datei bleibt dabei auf Ihrem Rechner.

Halten Sie die Vorführung kurz (unter einer Minute) und bereiten Sie vor, was Sie eintippen.

## Gemeinsamer Abschluss: von der Frage zur Antwort

Nach Gruppe E erklärt das Plenum gemeinsam den gesamten Weg. Jede Gruppe steuert ihre Station in **einem Satz** bei:

```
Frage "hello guppy"
   │
   ▼  A  Tokens, x/y, Batch, Maske  (beim Training)
   ▼  C  Embedding + Position → Vektor
   ▼  D  6 Blöcke, jeder addiert seinen Beitrag
   ▼  E  Logits → Temperatur → nächstes Token → anhängen …
Antwort "hi there. i just found a nice spot …"

   ↺  B  Beim Training: Verlust → Gradienten → Gewichte ändern
```

## Woran ein guter Beitrag zu erkennen ist

- Die Kernaussage ist in einem Satz verständlich, auch für Leute, die das Blatt nicht kennen.
- Jede Behauptung hat einen Beleg aus dem Notebook (Messwert, Ausgabe, Codezeile).
- Überraschungen werden genannt und nicht versteckt.
- Die Zeit wird eingehalten.

---

*Teil der entschärften Arbeitsblätter zu GuppyLM. Grundlage sind die Unterrichtsmaterialien aus [iludis52/DQR6_Material](https://github.com/iludis52/DQR6_Material) (Unterrichtsskizze Transformer, Sitzung 2).*
