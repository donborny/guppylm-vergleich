/* GuppyLM im Browser: Byte-Level-BPE-Tokenizer, Vorwärtslauf mit KV-Cache, Sampling.
   Nachbau von GuppyLM/antworte() aus guppylm_lehrnotebook_b.ipynb – ohne Bibliotheken.
   Gewichte kommen aus Hilfsskripte/exportiere_web.py (web/modelle/*.bin + *.json). */
"use strict";

(function (global) {
  // ------------------------------------------------------------------ Tokenizer

  // GPT-2-Zuordnung Byte -> druckbares Unicode-Zeichen (wie tokenizers ByteLevel).
  function byteZuZeichen() {
    const bs = [];
    for (let b = 33; b <= 126; b++) bs.push(b);
    for (let b = 161; b <= 172; b++) bs.push(b);
    for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    }
    const hin = new Array(256), zurueck = new Map();
    bs.forEach((b, i) => { hin[b] = String.fromCodePoint(cs[i]); zurueck.set(hin[b], b); });
    return { hin, zurueck };
  }
  const BYTES = byteZuZeichen();
  const GPT2_MUSTER = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  class Tokenizer {
    constructor(daten) {
      this.vocab = new Map(Object.entries(daten.vocab));
      this.idZuToken = [];
      for (const [t, id] of this.vocab) this.idZuToken[id] = t;
      this.rang = new Map(daten.merges.map((m, i) => [m, i]));
      this.sonder = new Map(Object.entries(daten.sondertokens));
      this.sonderIds = new Set(this.sonder.values());
      // längste Sondertokens zuerst prüfen
      this.sonderListe = [...this.sonder.keys()].sort((a, b) => b.length - a.length);
      this.cache = new Map();
      this.utf8 = new TextEncoder();
      this.sha256 = daten.sha256;
    }

    /** Aus tokenizer.to_str() der Python-Bibliothek tokenizers (steckt im Checkpoint). */
    static ausHfJson(text, sha256 = null) {
      const t = typeof text === "string" ? JSON.parse(text) : text;
      const art = x => x?.type ?? "keiner";
      if (art(t.model) !== "BPE") throw new Error(`Tokenizer-Modell ${art(t.model)} wird nicht unterstützt (erwartet BPE).`);
      if (art(t.pre_tokenizer) !== "ByteLevel" || t.pre_tokenizer.add_prefix_space || t.pre_tokenizer.use_regex === false) {
        throw new Error("Nur der Byte-Level-Tokenizer aus dem Kurs-Notebook wird unterstützt (pre_tokenizer ByteLevel, add_prefix_space=False).");
      }
      if (t.normalizer) throw new Error("Tokenizer mit Normalizer wird nicht unterstützt.");
      if (art(t.decoder) !== "ByteLevel") throw new Error("Tokenizer-Decoder muss ByteLevel sein.");
      return new Tokenizer({
        sha256,
        vocab: t.model.vocab,
        merges: t.model.merges.map(m => typeof m === "string" ? m : m.join(" ")),
        sondertokens: Object.fromEntries((t.added_tokens || []).map(a => [a.content, a.id])),
      });
    }

    tokenId(name) { return this.sonder.get(name) ?? this.vocab.get(name); }

    bpe(wort) {
      const gemerkt = this.cache.get(wort);
      if (gemerkt) return gemerkt;
      let teile = Array.from(wort);
      while (teile.length > 1) {
        let besterRang = Infinity, beste = -1;
        for (let i = 0; i < teile.length - 1; i++) {
          const r = this.rang.get(teile[i] + " " + teile[i + 1]);
          if (r !== undefined && r < besterRang) { besterRang = r; beste = i; }
        }
        if (beste < 0) break;
        const a = teile[beste], b = teile[beste + 1], neu = [];
        for (let i = 0; i < teile.length; i++) {
          if (i < teile.length - 1 && teile[i] === a && teile[i + 1] === b) { neu.push(a + b); i++; }
          else neu.push(teile[i]);
        }
        teile = neu;
      }
      const ids = teile.map(t => {
        const id = this.vocab.get(t);
        if (id === undefined) throw new Error(`Token ${JSON.stringify(t)} fehlt im Vokabular.`);
        return id;
      });
      this.cache.set(wort, ids);
      return ids;
    }

    kodiereText(text, ids) {
      for (const m of text.matchAll(GPT2_MUSTER)) {
        let wort = "";
        for (const b of this.utf8.encode(m[0])) wort += BYTES.hin[b];
        for (const id of this.bpe(wort)) ids.push(id);
      }
    }

    encode(text) {
      const ids = [];
      let rest = text;
      while (rest.length) {
        let pos = -1, treffer = null;
        for (const s of this.sonderListe) {
          const p = rest.indexOf(s);
          if (p >= 0 && (pos < 0 || p < pos)) { pos = p; treffer = s; }
        }
        if (pos < 0) { this.kodiereText(rest, ids); break; }
        if (pos > 0) this.kodiereText(rest.slice(0, pos), ids);
        ids.push(this.sonder.get(treffer));
        rest = rest.slice(pos + treffer.length);
      }
      return ids;
    }

    decode(ids, ohneSonder = true) {
      const bytes = [];
      let text = "";
      const leeren = () => { if (bytes.length) { text += new TextDecoder().decode(new Uint8Array(bytes)); bytes.length = 0; } };
      for (const id of ids) {
        const t = this.idZuToken[id];
        if (t === undefined) continue;
        if (this.sonderIds.has(id)) { if (!ohneSonder) { leeren(); text += t; } continue; }
        for (const z of t) bytes.push(BYTES.zurueck.get(z));
      }
      leeren();
      return text;
    }
  }

  // ------------------------------------------------------------------ .pt-Datei lesen
  // torch.save schreibt ein unkomprimiertes ZIP: <name>/data.pkl (Pickle mit der Struktur)
  // und <name>/data/<n> (Rohdaten je Storage). Der Pickle-Leser hier kennt nur die
  // Datentypen eines Checkpoints und führt nie Code aus der Datei aus.

  function leseZip(puffer) {
    const dv = new DataView(puffer), n = puffer.byteLength;
    let eocd = -1;
    for (let i = n - 22; i >= Math.max(0, n - 22 - 65535); i--) {
      if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) {
      throw new Error("Die Datei ist keine .pt-Datei im aktuellen PyTorch-Format (kein ZIP). " +
        "Wurde sie mit torch.save(...) aus dem Notebook gespeichert?");
    }
    const anzahl = dv.getUint16(eocd + 10, true), cdStart = dv.getUint32(eocd + 16, true);
    if (anzahl === 0xffff || cdStart === 0xffffffff) throw new Error("ZIP64-Dateien (über 4 GB) werden nicht unterstützt.");
    const eintraege = new Map(), dec = new TextDecoder();
    let p = cdStart;
    for (let i = 0; i < anzahl; i++) {
      if (dv.getUint32(p, true) !== 0x02014b50) throw new Error("ZIP-Verzeichnis beschädigt.");
      const methode = dv.getUint16(p + 10, true), groesse = dv.getUint32(p + 20, true);
      const nl = dv.getUint16(p + 28, true), el = dv.getUint16(p + 30, true), kl = dv.getUint16(p + 32, true);
      const lokal = dv.getUint32(p + 42, true);
      const name = dec.decode(new Uint8Array(puffer, p + 46, nl));
      const start = lokal + 30 + dv.getUint16(lokal + 26, true) + dv.getUint16(lokal + 28, true);
      eintraege.set(name, { methode, start, groesse });
      p += 46 + nl + el + kl;
    }
    return eintraege;
  }

  const DTYPES = {
    "torch FloatStorage": "float32", "torch HalfStorage": "float16",
    "torch BFloat16Storage": "bfloat16", "torch DoubleStorage": "float64",
  };

  function zuFloat32(puffer, start, dtype, anzahl) {
    switch (dtype) {
      case "float32": return new Float32Array(puffer.slice(start, start + 4 * anzahl));
      case "float64": return Float32Array.from(new Float64Array(puffer.slice(start, start + 8 * anzahl)));
      case "float16": return halbZuFloat(new Uint16Array(puffer.slice(start, start + 2 * anzahl)));
      case "bfloat16": {
        const u = new Uint16Array(puffer.slice(start, start + 2 * anzahl)), aus = new Float32Array(anzahl), a32 = new Uint32Array(aus.buffer);
        for (let i = 0; i < anzahl; i++) a32[i] = u[i] << 16;
        return aus;
      }
    }
    throw new Error(`Datentyp ${dtype} wird nicht unterstützt.`);
  }

  function lesePt(puffer) {
    const zip = leseZip(puffer);
    const pklName = [...zip.keys()].find(n => n.endsWith("/data.pkl") || n === "data.pkl");
    if (!pklName) throw new Error("In der .pt-Datei fehlt data.pkl.");
    const praefix = pklName.slice(0, -"data.pkl".length);
    const bo = zip.get(praefix + "byteorder");
    if (bo && new TextDecoder().decode(new Uint8Array(puffer, bo.start, bo.groesse)).trim() !== "little") {
      throw new Error("Big-Endian-Checkpoints werden nicht unterstützt.");
    }
    const pkl = zip.get(pklName);
    if (pkl.methode !== 0) throw new Error("Die .pt-Datei ist komprimiert; bitte die unveränderte Datei aus torch.save hochladen.");
    const bytes = new Uint8Array(puffer, pkl.start, pkl.groesse), dv = new DataView(puffer, pkl.start, pkl.groesse);
    const dec = new TextDecoder();

    const storageDaten = (key, dtype) => {
      const e = zip.get(`${praefix}data/${key}`);
      if (!e) throw new Error(`Tensordaten ${key} fehlen in der .pt-Datei.`);
      if (e.methode !== 0) throw new Error("Komprimierte Tensordaten werden nicht unterstützt.");
      return { start: e.start, groesse: e.groesse, dtype };
    };
    const MARK = Symbol("mark");
    const globals = {
      "collections OrderedDict": { aufruf: () => ({}) },
      "torch._utils _rebuild_tensor_v2": {
        aufruf: ([storage, offset, form, schritte]) => {
          const anzahl = form.reduce((a, b) => a * b, 1);
          // Zeilenweise (contiguous) gespeichert? Sonst lässt sich die Form nicht direkt übernehmen.
          let erwartet = 1;
          for (let i = form.length - 1; i >= 0; i--) {
            if (form[i] > 1 && schritte[i] !== erwartet) throw new Error("Nicht zusammenhängend gespeicherte Tensoren werden nicht unterstützt.");
            erwartet *= form[i];
          }
          const groesse = { float32: 4, float64: 8, float16: 2, bfloat16: 2 }[storage.dtype];
          return { __tensor: true, form: [...form], dtype: storage.dtype,
            daten: () => zuFloat32(puffer, storage.start + offset * groesse, storage.dtype, anzahl) };
        },
      },
      "torch._utils _rebuild_parameter": { aufruf: ([daten]) => daten },
    };
    for (const name of Object.keys(DTYPES)) globals[name] = { dtype: DTYPES[name] };

    const stapel = [], memo = new Map();
    const bisMark = () => { const i = stapel.lastIndexOf(MARK); const teil = stapel.splice(i); teil.shift(); return teil; };
    const setze = (obj, k, v) => { if (obj && typeof obj === "object") obj[k] = v; };
    let p = 0;
    const u8 = () => bytes[p++], u16 = () => { const v = dv.getUint16(p, true); p += 2; return v; };
    const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
    const text = n => { const s = dec.decode(bytes.subarray(p, p + n)); p += n; return s; };
    const zeile = () => { const e = bytes.indexOf(10, p); const s = dec.decode(bytes.subarray(p, e)); p = e + 1; return s; };
    const aufrufen = (f, args) => {
      if (!f || !f.aufruf) throw new Error(`Nicht unterstützter Inhalt in der .pt-Datei: ${f?.name ?? "unbekannte Funktion"}.`);
      return f.aufruf(args);
    };
    const holeGlobal = name => globals[name] || { name };

    for (;;) {
      const op = u8();
      switch (op) {
        case 0x80: p++; break;                                      // PROTO
        case 0x95: p += 8; break;                                   // FRAME
        case 0x2e: return stapel.pop();                             // STOP
        case 0x28: stapel.push(MARK); break;                        // MARK
        case 0x7d: stapel.push({}); break;                          // EMPTY_DICT
        case 0x5d: stapel.push([]); break;                          // EMPTY_LIST
        case 0x29: stapel.push([]); break;                          // EMPTY_TUPLE
        case 0x74: stapel.push(bisMark()); break;                   // TUPLE
        case 0x85: stapel.push([stapel.pop()]); break;              // TUPLE1
        case 0x86: { const b = stapel.pop(), a = stapel.pop(); stapel.push([a, b]); break; }
        case 0x87: { const c = stapel.pop(), b = stapel.pop(), a = stapel.pop(); stapel.push([a, b, c]); break; }
        case 0x58: stapel.push(text(u32())); break;                 // BINUNICODE
        case 0x8c: stapel.push(text(u8())); break;                  // SHORT_BINUNICODE
        case 0x8d: { const n = Number(dv.getBigUint64(p, true)); p += 8; stapel.push(text(n)); break; }
        case 0x55: stapel.push(text(u8())); break;                  // SHORT_BINSTRING
        case 0x54: stapel.push(text(u32())); break;                 // BINSTRING
        case 0x43: { const n = u8(); stapel.push(bytes.slice(p, p + n)); p += n; break; }
        case 0x42: { const n = u32(); stapel.push(bytes.slice(p, p + n)); p += n; break; }
        case 0x4a: { const v = dv.getInt32(p, true); p += 4; stapel.push(v); break; }   // BININT
        case 0x4b: stapel.push(u8()); break;                        // BININT1
        case 0x4d: stapel.push(u16()); break;                       // BININT2
        case 0x8a: {                                                // LONG1
          const n = u8(); let v = 0n;
          for (let i = n - 1; i >= 0; i--) v = (v << 8n) | BigInt(bytes[p + i]);
          if (n && bytes[p + n - 1] & 0x80) v -= 1n << BigInt(8 * n);
          p += n; stapel.push(Number(v)); break;
        }
        case 0x47: { const v = dv.getFloat64(p, false); p += 8; stapel.push(v); break; } // BINFLOAT
        case 0x4e: stapel.push(null); break;                        // NONE
        case 0x88: stapel.push(true); break;                        // NEWTRUE
        case 0x89: stapel.push(false); break;                       // NEWFALSE
        case 0x71: memo.set(u8(), stapel.at(-1)); break;            // BINPUT
        case 0x72: memo.set(u32(), stapel.at(-1)); break;           // LONG_BINPUT
        case 0x94: memo.set(memo.size, stapel.at(-1)); break;       // MEMOIZE
        case 0x68: stapel.push(memo.get(u8())); break;              // BINGET
        case 0x6a: stapel.push(memo.get(u32())); break;             // LONG_BINGET
        case 0x63: { const m = zeile(), n = zeile(); stapel.push(holeGlobal(`${m} ${n}`)); break; }  // GLOBAL
        case 0x93: { const n = stapel.pop(), m = stapel.pop(); stapel.push(holeGlobal(`${m} ${n}`)); break; }
        case 0x52: { const args = stapel.pop(), f = stapel.pop(); stapel.push(aufrufen(f, args)); break; } // REDUCE
        case 0x81: { const args = stapel.pop(), f = stapel.pop(); stapel.push(aufrufen(f, args)); break; } // NEWOBJ
        case 0x62: stapel.pop(); break;                             // BUILD (Metadaten des state_dict ignorieren)
        case 0x73: { const v = stapel.pop(), k = stapel.pop(); setze(stapel.at(-1), k, v); break; }       // SETITEM
        case 0x75: { const t = bisMark(), d = stapel.at(-1); for (let i = 0; i < t.length; i += 2) setze(d, t[i], t[i + 1]); break; }
        case 0x61: { const v = stapel.pop(); stapel.at(-1).push(v); break; }                               // APPEND
        case 0x65: { const t = bisMark(); stapel.at(-1).push(...t); break; }                               // APPENDS
        case 0x51: {                                                // BINPERSID
          const [art, typ, key, , ] = stapel.pop();
          if (art !== "storage" || !typ?.dtype) throw new Error(`Nicht unterstützter Speichertyp ${typ?.name ?? art} in der .pt-Datei.`);
          stapel.push(storageDaten(key, typ.dtype));
          break;
        }
        default:
          throw new Error(`Unbekannter Pickle-Befehl 0x${op.toString(16)} an Position ${p - 1}.`);
      }
    }
  }

  // ------------------------------------------------------------------ Modell

  function halbZuFloat(u16) {
    const aus = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      aus[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return aus;
  }

  const KONFIG_FELDER = ["vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads", "ffn_hidden", "pad_id", "bos_id", "eos_id"];

  /** Erwartete Tensorformen der GuppyLM-Architektur aus dem Notebook. */
  function erwarteteFormen(k) {
    const C = k.d_model, F = k.ffn_hidden, f = { "tok_emb.weight": [k.vocab_size, C], "pos_emb.weight": [k.max_seq_len, C] };
    for (let i = 0; i < k.n_layers; i++) {
      const B = `blocks.${i}.`;
      Object.assign(f, {
        [B + "norm1.weight"]: [C], [B + "norm1.bias"]: [C],
        [B + "attn.qkv.weight"]: [3 * C, C], [B + "attn.qkv.bias"]: [3 * C],
        [B + "attn.out.weight"]: [C, C], [B + "attn.out.bias"]: [C],
        [B + "norm2.weight"]: [C], [B + "norm2.bias"]: [C],
        [B + "ffn.hoch.weight"]: [F, C], [B + "ffn.hoch.bias"]: [F],
        [B + "ffn.runter.weight"]: [C, F], [B + "ffn.runter.bias"]: [C],
      });
    }
    f["norm.weight"] = [C]; f["norm.bias"] = [C];
    return f;
  }

  class GuppyModell {
    /** gewichte: {name: Float32Array}, formen: {name: [..]} (optional, zur Prüfung). */
    constructor(meta, gewichte, tokenizer, formen = null) {
      this.meta = meta;
      this.k = meta.konfig;
      this.tok = tokenizer;
      const k = this.k;
      for (const feld of KONFIG_FELDER) {
        if (!Number.isInteger(k?.[feld])) throw new Error(`Konfiguration unvollständig: ${feld} fehlt.`);
      }
      if (k.d_model % k.n_heads) throw new Error("d_model ist nicht durch n_heads teilbar.");
      if (tokenizer.vocab.size !== k.vocab_size) throw new Error(`Tokenizer (${tokenizer.vocab.size} Tokens) und Modell (vocab_size ${k.vocab_size}) passen nicht zusammen.`);
      for (const [name, id] of [["<pad>", k.pad_id], ["<|im_start|>", k.bos_id], ["<|im_end|>", k.eos_id]]) {
        if (tokenizer.tokenId(name) !== id) throw new Error(`Sondertoken ${name} passt nicht zur Modellkonfiguration.`);
      }
      const soll = erwarteteFormen(k);
      for (const [name, form] of Object.entries(soll)) {
        const w = gewichte[name];
        if (!w) throw new Error(`Gewicht ${name} fehlt. Ist das ein GuppyLM-Transformer aus dem Kurs-Notebook?`);
        const n = form.reduce((a, b) => a * b, 1);
        const ist = formen?.[name];
        if (w.length !== n || (ist && ist.join("x") !== form.join("x"))) {
          throw new Error(`Gewicht ${name} hat Form ${ist ? ist.join("×") : w.length}, erwartet ${form.join("×")} (laut Konfiguration).`);
        }
      }
      if (gewichte["lm_head.weight"] && gewichte["lm_head.weight"].length !== k.vocab_size * k.d_model) {
        throw new Error("lm_head.weight hat die falsche Form.");
      }
      this.w = gewichte;
      const C = k.d_model;
      this.puffer = { x: new Float32Array(C), h: new Float32Array(C), qkv: new Float32Array(3 * C),
        att: new Float32Array(C), o: new Float32Array(C), f: new Float32Array(this.k.ffn_hidden),
        s: new Float32Array(this.k.max_seq_len), logits: new Float32Array(this.k.vocab_size) };
    }

    /** Aus den Dateien von exportiere_web.py (.json-Metadaten + .bin-Puffer). */
    static ausExport(meta, puffer, tokenizer) {
      if (puffer.byteLength !== meta.bytes) throw new Error(`Gewichtsdatei hat ${puffer.byteLength} statt ${meta.bytes} Bytes.`);
      const w = {}, formen = {};
      for (const [name, t] of Object.entries(meta.tensoren)) {
        w[name] = meta.dtype === "float16"
          ? halbZuFloat(new Uint16Array(puffer, t.offset, t.anzahl))
          : new Float32Array(puffer.slice(t.offset, t.offset + 4 * t.anzahl));
        formen[name] = t.form;
      }
      return new GuppyModell(meta, w, tokenizer, formen);
    }

    /** Direkt aus der .pt-Datei, die das Notebook speichert. */
    static async ausPt(puffer, dateiname = "modell.pt", geaendert = null) {
      const ckpt = lesePt(puffer);
      if (!ckpt || typeof ckpt !== "object") throw new Error("Die Datei enthält keinen Checkpoint (kein Dictionary).");
      if (!ckpt.model_state_dict) {
        throw new Error("Kein Checkpoint aus dem Kurs-Notebook: Feld model_state_dict fehlt. " +
          "Bitte die Datei hochladen, die das Notebook mit torch.save(checkpoint, …) speichert (z. B. bestes_modell_b.pt).");
      }
      if (!ckpt.konfig) throw new Error("Feld konfig fehlt im Checkpoint.");
      if (typeof ckpt.tokenizer_json !== "string") {
        throw new Error("Alter Checkpoint ohne zugeordneten Tokenizer (tokenizer_json fehlt). Bitte mit dem aktuellen Notebook neu trainieren.");
      }
      if (ckpt.tokenizer_sha256 && globalThis.crypto?.subtle) {
        const roh = new TextEncoder().encode(ckpt.tokenizer_json);
        const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", roh))].map(b => b.toString(16).padStart(2, "0")).join("");
        if (hash !== ckpt.tokenizer_sha256) throw new Error("Tokenizer-Prüfsumme im Checkpoint stimmt nicht.");
      }
      const tok = Tokenizer.ausHfJson(ckpt.tokenizer_json, ckpt.tokenizer_sha256 ?? null);
      const w = {}, formen = {};
      let parameter = 0;
      for (const [name, t] of Object.entries(ckpt.model_state_dict)) {
        if (!t || !t.__tensor) continue;
        w[name] = t.daten(); formen[name] = t.form;
        if (name !== "lm_head.weight") parameter += w[name].length;
      }
      const meta = {
        name: dateiname.replace(/\.pt$/i, ""), quelle: dateiname,
        schritt: ckpt.schritt ?? null, eval_verlust: ckpt.eval_verlust ?? null,
        konfig: ckpt.konfig, parameter, tokenizer_sha256: ckpt.tokenizer_sha256 ?? null,
        gespeichert: geaendert,
      };
      return new GuppyModell(meta, w, tok, formen);
    }

    static linear(W, b, x, y, nIn, nOut) {
      for (let o = 0; o < nOut; o++) {
        let s = b ? b[o] : 0;
        const z = o * nIn;
        for (let i = 0; i < nIn; i++) s += W[z + i] * x[i];
        y[o] = s;
      }
    }

    static layerNorm(x, g, b, y) {
      const n = x.length;
      let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
      let v = 0; for (let i = 0; i < n; i++) { const d = x[i] - m; v += d * d; } v /= n;
      const r = 1 / Math.sqrt(v + 1e-5);
      for (let i = 0; i < n; i++) y[i] = (x[i] - m) * r * g[i] + b[i];
    }

    neuerCache() {
      const { n_layers, max_seq_len, d_model } = this.k;
      return { laenge: 0, k: Array.from({ length: n_layers }, () => new Float32Array(max_seq_len * d_model)),
        v: Array.from({ length: n_layers }, () => new Float32Array(max_seq_len * d_model)) };
    }

    /** Ein Token an Position cache.laenge verarbeiten; liefert Logits (wiederverwendeter Puffer). */
    schritt(id, cache) {
      const { d_model: C, n_heads: H, n_layers: L, ffn_hidden: F, vocab_size: V, max_seq_len } = this.k;
      const hd = C / H, pos = cache.laenge, w = this.w, p = this.puffer, lin = GuppyModell.linear;
      if (pos >= max_seq_len) throw new Error("Kontextgrenze überschritten.");
      const { x, h, qkv, att, o, f, s, logits } = p;
      const te = w["tok_emb.weight"], pe = w["pos_emb.weight"];
      for (let i = 0; i < C; i++) x[i] = te[id * C + i] + pe[pos * C + i];
      const skala = 1 / Math.sqrt(hd);

      for (let l = 0; l < L; l++) {
        const B = `blocks.${l}.`;
        GuppyModell.layerNorm(x, w[B + "norm1.weight"], w[B + "norm1.bias"], h);
        lin(w[B + "attn.qkv.weight"], w[B + "attn.qkv.bias"], h, qkv, C, 3 * C);
        const K = cache.k[l], Vc = cache.v[l];
        K.set(qkv.subarray(C, 2 * C), pos * C);
        Vc.set(qkv.subarray(2 * C, 3 * C), pos * C);
        for (let kopf = 0; kopf < H; kopf++) {
          const a = kopf * hd;
          let max = -Infinity;
          for (let t = 0; t <= pos; t++) {
            let d = 0; const z = t * C + a;
            for (let j = 0; j < hd; j++) d += qkv[a + j] * K[z + j];
            s[t] = d * skala; if (s[t] > max) max = s[t];
          }
          let summe = 0;
          for (let t = 0; t <= pos; t++) { s[t] = Math.exp(s[t] - max); summe += s[t]; }
          for (let j = 0; j < hd; j++) att[a + j] = 0;
          for (let t = 0; t <= pos; t++) {
            const g = s[t] / summe, z = t * C + a;
            for (let j = 0; j < hd; j++) att[a + j] += g * Vc[z + j];
          }
        }
        lin(w[B + "attn.out.weight"], w[B + "attn.out.bias"], att, o, C, C);
        for (let i = 0; i < C; i++) x[i] += o[i];

        GuppyModell.layerNorm(x, w[B + "norm2.weight"], w[B + "norm2.bias"], h);
        lin(w[B + "ffn.hoch.weight"], w[B + "ffn.hoch.bias"], h, f, C, F);
        for (let i = 0; i < F; i++) {   // GELU, tanh-Näherung
          const u = f[i];
          f[i] = 0.5 * u * (1 + Math.tanh(0.7978845608028654 * (u + 0.044715 * u * u * u)));
        }
        lin(w[B + "ffn.runter.weight"], w[B + "ffn.runter.bias"], f, o, F, C);
        for (let i = 0; i < C; i++) x[i] += o[i];
      }
      GuppyModell.layerNorm(x, w["norm.weight"], w["norm.bias"], h);
      lin(w["lm_head.weight"] || te, null, h, logits, C, V);   // im Notebook: lm_head = tok_emb (weight tying)
      cache.laenge = pos + 1;
      return logits;
    }
  }

  // ------------------------------------------------------------------ Sampling

  // Kleiner, seedbarer Zufallsgenerator (mulberry32). Gleicher Seed -> gleiche Zahlenfolge.
  function zufall(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function waehle(logits, k, temperatur, topK, rnd) {
    const l = Float64Array.from(logits);
    l[k.pad_id] = -Infinity; l[k.bos_id] = -Infinity;
    if (temperatur === 0) {
      let best = 0;
      for (let i = 1; i < l.length; i++) if (l[i] > l[best]) best = i;
      return best;
    }
    for (let i = 0; i < l.length; i++) l[i] /= temperatur;
    if (topK) {
      const grenze = Float64Array.from(l).sort().reverse()[Math.min(topK, l.length) - 1];
      for (let i = 0; i < l.length; i++) if (l[i] < grenze) l[i] = -Infinity;
    }
    let max = -Infinity; for (const v of l) if (v > max) max = v;
    let summe = 0;
    for (let i = 0; i < l.length; i++) { l[i] = Math.exp(l[i] - max); summe += l[i]; }
    let r = rnd() * summe;
    for (let i = 0; i < l.length; i++) { r -= l[i]; if (r < 0 && l[i] > 0) return i; }
    for (let i = l.length - 1; i >= 0; i--) if (l[i] > 0) return i;
    return 0;
  }

  /** Wie antworte() im Notebook. beiToken(text) wird nach jedem Token aufgerufen (für Live-Anzeige). */
  async function antworte(modell, frage, { temperatur = 0.7, topK = 50, maxNeu = 64, seed = null, beiToken = null } = {}) {
    const k = modell.k, tok = modell.tok;
    const ids = tok.encode(`<|im_start|>user\n${frage}<|im_end|>\n<|im_start|>assistant\n`);
    if (!ids.length || ids.length > k.max_seq_len) throw new Error(`Der Prompt benötigt 1 bis ${k.max_seq_len} Tokens.`);
    const rnd = seed === null ? Math.random : zufall(seed);
    const cache = modell.neuerCache();
    const t0 = (global.performance || Date).now();
    let logits;
    for (const id of ids) logits = modell.schritt(id, cache);
    const neu = [];
    let grund = "Tokenlimit";
    for (let n = 0; n < maxNeu; n++) {
      if (cache.laenge >= k.max_seq_len) { grund = "Kontextgrenze"; break; }
      const id = waehle(logits, k, temperatur, topK, rnd);
      neu.push(id);
      if (id === k.eos_id) { grund = "Endtoken"; break; }
      if (beiToken) { beiToken(tok.decode(neu).trim()); await new Promise(r => setTimeout(r, 0)); }
      if (n < maxNeu - 1) logits = modell.schritt(id, cache);
    }
    return { text: tok.decode(neu).trim(), grund, promptTokens: ids.length, neueTokens: neu.length,
      ms: Math.round((global.performance || Date).now() - t0) };
  }

  global.Guppy = { Tokenizer, GuppyModell, antworte, zufall, lesePt };
  if (typeof module !== "undefined") module.exports = global.Guppy;
})(typeof globalThis !== "undefined" ? globalThis : this);
