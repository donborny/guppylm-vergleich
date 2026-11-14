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

  // ------------------------------------------------------------------ Modell

  function halbZuFloat(u16) {
    const aus = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
      aus[i] = e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
    }
    return aus;
  }

  class GuppyModell {
    constructor(meta, puffer, tokenizer) {
      this.meta = meta;
      this.k = meta.konfig;
      this.tok = tokenizer;
      if (tokenizer.vocab.size !== this.k.vocab_size) throw new Error("Tokenizer und Modell haben verschiedene Vokabulargrößen.");
      for (const [name, id] of [["<pad>", this.k.pad_id], ["<|im_start|>", this.k.bos_id], ["<|im_end|>", this.k.eos_id]]) {
        if (tokenizer.tokenId(name) !== id) throw new Error(`Sondertoken ${name} passt nicht zur Modellkonfiguration.`);
      }
      if (puffer.byteLength !== meta.bytes) throw new Error(`Gewichtsdatei hat ${puffer.byteLength} statt ${meta.bytes} Bytes.`);
      this.w = {};
      for (const [name, t] of Object.entries(meta.tensoren)) {
        this.w[name] = meta.dtype === "float16"
          ? halbZuFloat(new Uint16Array(puffer, t.offset, t.anzahl))
          : new Float32Array(puffer.slice(t.offset, t.offset + 4 * t.anzahl));
      }
      const C = this.k.d_model;
      this.puffer = { x: new Float32Array(C), h: new Float32Array(C), qkv: new Float32Array(3 * C),
        att: new Float32Array(C), o: new Float32Array(C), f: new Float32Array(this.k.ffn_hidden),
        s: new Float32Array(this.k.max_seq_len), logits: new Float32Array(this.k.vocab_size) };
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
      lin(te, null, h, logits, C, V);          // lm_head = tok_emb (weight tying)
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

  global.Guppy = { Tokenizer, GuppyModell, antworte, zufall };
  if (typeof module !== "undefined") module.exports = global.Guppy;
})(typeof globalThis !== "undefined" ? globalThis : this);
