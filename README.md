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

## Neuen Checkpoint veröffentlichen

Im Kursordner (neben dem Notebook):

```bash
python Hilfsskripte/exportiere_web.py              # alle Checkpoints/*.pt
python Hilfsskripte/exportiere_web.py Checkpoints/mein_modell.pt   # nur einer
cd web
git add modelle && git commit -m "Checkpoint aktualisiert" && git push
```

GitHub Pages zeigt die Änderung nach ein bis zwei Minuten. Um einen Checkpoint zu entfernen, löschst du seine `.json` und `.bin` in `modelle/` und führst den Export erneut aus, damit `index.json` neu geschrieben wird.

Einen Checkpoint ohne Veröffentlichung testen: exportieren und auf der Seite unter „Eigenen Checkpoint ausprobieren“ die `.json`, die `.bin` und die `tokenizer-….json` auswählen.

## Lokal starten

```bash
cd web && python -m http.server 8000   # dann http://localhost:8000
```

Direkt per Doppelklick geöffnet (`file://`) blockiert der Browser das Laden der Modelldateien.

## Genauigkeit

Geprüft gegen PyTorch: Der Tokenizer kodiert 3008 von 3008 Testtexten identisch. Mit Temperatur 0 liefern beide Checkpoints bei 20 von 20 Fragen dieselbe Antwort wie das Notebook, auch mit den float16-Gewichten. Mit Temperatur über 0 kommen die Zufallszahlen aus einem eigenen Generator (mulberry32). Antworten mit Seed sind daher reproduzierbar, aber nicht identisch mit `torch.multinomial`.
