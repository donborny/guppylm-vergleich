# GuppyLM · Checkpoint-Vergleich

Zwei Trainingsstände eines kleinen Transformer-Sprachmodells (GuppyLM, ca. 11,7 Mio. Parameter) nebeneinander befragen. Alles läuft im Browser: Tokenizer, Vorwärtslauf und Sampling sind in [`guppy.js`](guppy.js) ohne Bibliotheken nachgebaut. Es gibt keinen Server und keine Datenübertragung.

GuppyLM ist ein Lehrmodell aus einem Kurs über Transformer. Es spielt einen Goldfisch und antwortet auf Englisch.

## Dateien

| Datei | Inhalt |
|---|---|
| `index.html` | Oberfläche |
| `guppy.js` | Byte-Level-BPE-Tokenizer, Pre-Norm-Decoder mit KV-Cache, Temperatur-/Top-k-Sampling mit seedbarem Zufall |
| `modelle/index.json` | Liste der angebotenen Checkpoints |
| `modelle/<name>.json` / `.bin` | Konfiguration und Gewichte (float16) eines Checkpoints |
| `modelle/tokenizer-<sha>.json` | Vokabular und Merges |

## Eigenes Modell testen (für Kursteilnehmende)

Die `.pt`-Datei aus dem Notebook (z. B. `Checkpoints/bestes_modell_b.pt`) auf die Seite ziehen oder über „Datei auswählen …“ laden. Ein Export oder Python ist dafür nicht nötig. Die Datei verlässt den Rechner nicht: `guppy.js` liest das PyTorch-Format (ZIP und Pickle) direkt im Browser und führt dabei keinen Code aus der Datei aus.

Voraussetzungen an die Datei, wie sie das Kurs-Notebook erzeugt:

- gespeichert mit `torch.save({...})` als Dictionary mit `model_state_dict`, `konfig` und `tokenizer_json` (`schritt`, `eval_verlust` und `tokenizer_sha256` sind optional)
- GuppyLM-Architektur: Pre-Norm-Blöcke mit `attn.qkv`/`attn.out` und `ffn.hoch`/`ffn.runter`, GELU (tanh)
- Größe frei wählbar: `n_layers`, `d_model`, `n_heads`, `ffn_hidden`, `max_seq_len` und Vokabular kommen aus `konfig`
- Gewichte in float32, float16 oder bfloat16, `lm_head` gebunden oder eigen
- Byte-Level-BPE-Tokenizer aus `tokenizers`

Passt etwas nicht, nennt die Seite den Grund, zum Beispiel welcher Tensor welche Form hat.

## Neuen Checkpoint für alle veröffentlichen

Im Kursordner (neben dem Notebook):

```bash
python Hilfsskripte/exportiere_web.py              # alle Checkpoints/*.pt
python Hilfsskripte/exportiere_web.py Checkpoints/mein_modell.pt   # nur einer
cd web
git add modelle && git commit -m "Checkpoint aktualisiert" && git push
```

GitHub Pages zeigt die Änderung nach ein bis zwei Minuten. Um einen Checkpoint zu entfernen, löschst du seine `.json` und `.bin` in `modelle/` und führst den Export erneut aus, damit `index.json` neu geschrieben wird.

## Lokal starten

```bash
cd web && python -m http.server 8000   # dann http://localhost:8000
```

Direkt per Doppelklick geöffnet (`file://`) blockiert der Browser das Laden der Modelldateien.

## Genauigkeit

Geprüft gegen PyTorch: Der Tokenizer kodiert 3008 von 3008 Testtexten identisch. Mit Temperatur 0 liefern beide Checkpoints bei 20 von 20 Fragen dieselbe Antwort wie das Notebook, auch mit den float16-Gewichten. Mit Temperatur über 0 kommen die Zufallszahlen aus einem eigenen Generator (mulberry32). Antworten mit Seed sind daher reproduzierbar, aber nicht identisch mit `torch.multinomial`.

## Unterrichtsmaterial: entschärfte Arbeitsblätter

Im Ordner [`arbeitsblaetter/`](arbeitsblaetter/) liegen fünf Arbeitsblätter für die Gruppenphase (75 Minuten, je eine Vierergruppe) und eine Anleitung fürs Plenum:

| Gruppe | Thema |
|---|---|
| [A](arbeitsblaetter/Arbeitsblatt_A_Trainingsbatch.md) | Vom Dialog zum Trainingsbatch: x/y, Padding, kausale Maske |
| [B](arbeitsblaetter/Arbeitsblatt_B_Loss_und_Backprop.md) | Der Verlust und sein Weg zurück ins Netz |
| [C](arbeitsblaetter/Arbeitsblatt_C_Was_kann_das_Netz.md) | Was kann das Netz am Ende? Embeddings und Positionen |
| [D](arbeitsblaetter/Arbeitsblatt_D_Residualverbindung.md) | Die Residualverbindung |
| [E](arbeitsblaetter/Arbeitsblatt_E_Naechstes_Token.md) | Das nächste Token, Temperatur, Weight Tying |
| [Plenum](arbeitsblaetter/Anleitung_Plenum.md) | So stellen die Gruppen ihr Ergebnis vor (Reihenfolge A, C, D, B, E) |

Jedes Blatt hat denselben Aufbau: kurzer Erklärtext mit Grafik, Lesefragen, Verständnisfragen, ein fertiges Experiment für Variante b des Notebooks (ohne neues Training) und eine Vorgabe fürs Plenum.

Die Blätter sind eine bearbeitete, entschärfte Fassung der Arbeitsblätter aus [iludis52/DQR6_Material](https://github.com/iludis52/DQR6_Material). Dort liegen auch die Notebooks und Checkpoints.
