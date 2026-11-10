"""Target-aware prompt compiler (2.4).

A per-target table that drives generation against the *downstream* budget and
structure instead of a single opaque string:

| field | purpose |
|---|---|
| `system` | per-target system prompt (None -> the node's system_prompt field) |
| `budget` | downstream token budget the output must fit (None -> max_tokens) |
| `headers` | required section headers, constrained at line starts by the 2.3 mask |
| `post` | post-processing steps: strip_markdown / single_paragraph / tag_dedupe |

Outputs of the node become `(prompt, token_count, report)` —
values the next node can act on, instead of one opaque string.
"""
import re

DEFAULT_SYSTEM = None  # sentinel: use the node's system_prompt field / DEFAULT_SYSTEM

TARGETS = {
    "default": {
        "system": None,
        "budget": None,
        "headers": [],
        "post": [],
        "notes": "today's behaviour: fluid prose paragraph, DEFAULT_SYSTEM",
    },
    "structured": {
        "system": (
            "You are an expert image prompt generation system that answers in "
            "four labelled sections, one per line, in exactly this order:\n"
            "Subject: <the main subject, its colour, pose and condition>\n"
            "Scene: <setting, time of day, lighting and atmosphere>\n"
            "Details: <materials, textures, small visible elements>\n"
            "Style: <medium, rendering style, composition and mood>\n"
            "Expand the user's idea into the four sections. Use full, "
            "grammatically correct sentences. Output ONLY the four lines."
        ),
        "budget": 512,
        "headers": ["Subject", "Scene", "Details", "Style"],
        "post": ["strip_markdown"],
        "notes": "SDXL/Qwen-Image sectioned format; headers constrained by mask",
    },
    "tag": {
        "system": (
            "You are an expert image prompt generation system that outputs a "
            "comma-separated list of short descriptive tags, most important "
            "first, no sentences, no line breaks, no markdown. Tags describe "
            "subject, appearance, pose, setting, lighting, materials, style."
        ),
        "budget": 512,
        "headers": [],
        "post": ["tag_dedupe"],
        "notes": "SDXL tag-style (e.g. anime illustration) targets",
    },
}


def get_target(name):
    """Return the target table entry, or the default entry for unknown names."""
    return TARGETS.get(name) or TARGETS["default"]


# ------------------------------------------------------------------ postprocess

def _strip_markdown(text):
    text = re.sub(r"```(?:\w*)?\n?", "", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    return text.replace("**", "").replace("__", "")


def _single_paragraph(text):
    return " ".join(text.split()).strip()


def _tag_dedupe(text):
    seen, out = set(), []
    for t in text.split(","):
        t = t.strip()
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return ", ".join(out)


_APPLY = {"strip_markdown": _strip_markdown,
          "single_paragraph": _single_paragraph,
          "tag_dedupe": _tag_dedupe}


def postprocess(text, target):
    """Apply the target's post steps in table order."""
    for step in target.get("post") or []:
        if step in _APPLY:
            text = _APPLY[step](text)
    return text


# --------------------------------------------------------------- header mask

def build_header_mask(tok, headers, stop_ids, vocab, device):
    """Boolean [vocab] mask of the tokens allowed at a line start.

    Assembly is offline from the tokenizer: the first token of every header
    string plus every stop token (2.3 v1: headers first, then stop — nothing
    else).  Returns None when `headers` is empty (no constraint).
    """
    if not headers:
        return None
    import torch
    mask = torch.zeros(vocab, dtype=torch.bool, device=device)
    ids = set(stop_ids)
    for h in headers:
        enc = tok.encode(h)
        ids.update(enc.ids[:1] if hasattr(enc, "ids") else enc[:1])
    mask[torch.tensor(sorted(ids), dtype=torch.long, device=device)] = True
    return mask



# -------------------------------------------------------------- section grammar

class SectionGrammar:
    """Full section grammar (2.3 v2): the whole header, token by token.

    v1 constrained only the line's first token; v2 also constrains the
    continuation of a chosen header up to (and including) the colon — the
    allowed sets are built offline from the tokenizer, per-token cost is one
    mask lookup.  After the header's own token path completes, the body is
    free until a newline token re-enters the line-start state.  The stop
    tokens are always allowed (early exit).

    Headers are assumed to tokenise to uniform length as `encode(h + ":")`
    for the shared state (verified for the shipped targets); with
    non-uniform sets the grammar falls back to v1 masks.
    """

    def __init__(self, headers, tok, vocab, device, stop_ids=(), newline_id=None):
        def _ids(text):
            enc = tok.encode(text)
            return enc.ids if hasattr(enc, "ids") else list(enc)
        self.paths = [_ids(h + ":") for h in headers]
        self.vocab, self.device = vocab, device
        self.stop = set(stop_ids)
        self.newline_id = newline_id
        self._cache = {}
        lens = {len(p) for p in self.paths}
        self.uniform = len(lens) == 1 and bool(self.paths)

    def start_state(self):
        """Line-start state: candidate header indices + position within path."""
        return (frozenset(range(len(self.paths))), 0)

    def mask(self, state):
        """Boolean [vocab] mask for `state`, or None in free/body mode."""
        if state is None:
            return None
        cands, k = state
        allowed = {self.paths[i][k] for i in cands if k < len(self.paths[i])}
        allowed |= self.stop
        if k >= min(len(self.paths[i]) for i in cands):
            # header fully emitted: body is free
            return None
        key = (cands, k)
        m = self._cache.get(key)
        if m is None:
            import torch
            m = torch.zeros(self.vocab, dtype=torch.bool, device=self.device)
            m[torch.tensor(sorted(allowed), dtype=torch.long,
                           device=self.device)] = True
            self._cache[key] = m
        return m

    def advance(self, state, tok):
        """Feed the sampled token; return the next state (None = free body)."""
        if state is None:
            if self.newline_id is not None and tok == self.newline_id:
                return self.start_state()
            return None
        cands, k = state
        if tok in self.stop:
            return None
        if k == 0:
            nxt = frozenset(i for i in cands if self.paths[i][0] == tok)
            return (nxt, 1) if nxt else None
        nxt = frozenset(i for i in cands if self.paths[i][k] == tok)
        if not nxt:
            return None
        if k + 1 >= len(self.paths[next(iter(nxt))]):
            return None
        return (nxt, k + 1)


# ------------------------------------------------------- keyword preservation
#
# Content words of the user's idea are preserved (Phase 5).  Everything here
# is stem-tolerant and case-insensitive so the model can inflect freely
# (dance / dancing) while the *concept* survives.

_BROAD_STOPWORDS = set(
    """a an the and or nor of in on at for with to into from by as is are was were
    be been being it its this that these those you your we our they their he his
    she her i me my mine not no do does did done doing can could should would
    will just too very so such some any all each both other another many much
    few most more less also over under about above below off out up down per via
    with without beyond across between through until since while after before
    when where how what which who whom whose why than then there here where
    please make make it made let lets get gets got give gives take takes taking
    want need must may might shall like unlike same different new now one ones
    """.split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def _stem_word(w):
    """Multi-pass stemmer: strips 1-3 lag suffixes (dancing -> danc)."""
    w = w.lower()
    for _ in range(3):
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        else:
            break
    return w


def preserved_keywords(text):
    """Content words of `text` to preserve, in order of appearance.

    `[A-Za-z][A-Za-z'-]{2,}` (≥ 3 chars), broadened stopword list, deduped by
    stem, original case kept.  Hyphenated words are split into their parts
    (punct/hyphens are not treated as parts of stems) — `rain-soaked` becomes
    `rain`, `soaked` — so both are matched freely regardless of whether the
    model writes `rain soaked`, `rain-soaked` or `soaked by the rain`.
    """
    seen = set()
    out = []

    def _emit(part):
        s = _stem_word(part)
        lw = part.lower()
        if lw in _BROAD_STOPWORDS or len(lw) < 3 or s in seen or len(s) < 2:
            return
        seen.add(s)
        out.append(part)

    for m in WORD_RE.finditer(text):
        w = m.group(0)
        for part in w.split('-'):
            _emit(part)
    return out


def _stem(w):
    return _stem_word(w)


def stem_variants(kw):
    """Case-insensitive stem-tolerant alternatives for one preserved word.

    Covers regular morphology: bare stem, -s/-es plurals (also of words
    ending in -e: ``streets`` stems to ``street``), -ed/-ing forms, and -y
    plurals (``city`` -> ``cities``).
    """
    base = kw.lower()
    stem = _stem(base)
    cands = {base, stem, stem + "s", stem + "es", stem + "ed", stem + "ing",
             stem + "e" + "ing", stem + "e" + "ed"}
    if base.endswith("e"):
        cands.add(base[:-1] + "s")
        cands.add(base[:-1] + "ed")
        cands.add(base[:-1] + "ing")
    if base.endswith("y"):
        cands.add(base[:-1] + "ies")
        cands.add(base[:-1] + "ied")
    for suf in ("ing", "ed", "es", "s", "'s"):
        if base.endswith(suf):
            cands.add(base[: -len(suf)]) if suf != "'s" else cands.add(base[: -2])
    return {c for c in cands if c}


def present_keywords(text, keywords):
    """Which preserved keywords appear in `text` (case-insensitive, stem-tolerant).

    Hyphenated words in `text` split into their parts, so preserving
    `rain` and `soaked` both match a model's `rain-soaked` (in any order).
    """
    low = re.sub(r"[^a-z']+", " ", text.lower())
    toks = {w for t in low.split() for w in t.split('-')}
    hits, miss = [], []
    for kw in keywords:
        variants = stem_variants(kw)
        ok = any(v in toks or any(t.startswith(v) for t in toks)
                 for v in variants)
        (hits if ok else miss).append(kw)
    return hits, miss


def score_candidates(texts, keywords, budget=None, headers=None, tok=None,
                     require_all=False):
    """Deterministic candidate scores (2.2).

    keyword coverage 40 % + budget fit 25 % + structure 20 % + repetition
    15 %.  With `require_all=True` (Phase 5), a candidate missing *any*
    preserved stem scores 0.0 (disqualified); if every candidate misses, the
    best-coverage row wins instead and a flag is set on the returned report.

    Returns `(winner_index, per-candidate score dicts)` (plus a third element
    `report_flags` when `require_all` is set).
    """
    scores = []
    n = len(texts)
    for idx, text in enumerate(texts):
        kw_hits, kw_miss = present_keywords(text, keywords) if keywords else ([], [])
        kw = 1.0
        if keywords:
            kw = (len(kw_hits) / len(keywords))
        cnt = len(tok.encode(text)) if tok is not None else len(text.split())
        fit = 1.0 if not budget else min(1.0, budget / max(1, cnt))
        st = 1.0
        if headers:
            pos = [text.lower().find(h.lower() + ":") for h in headers]
            st = 1.0 if all(p >= 0 for p in pos) and pos == sorted(pos) else 0.0
        grams = re.findall(r"\S+ \S+ \S+ \S+",
                           re.sub(r"[^a-z]+", " ", text.lower()))
        dup = (len(grams) - len(set(grams)))
        rep = max(0.0, 1.0 - dup)
        total = 0.4 * kw + 0.25 * fit + 0.2 * st + 0.15 * rep
        scores.append({
            "index": idx,
            "score": round(total, 4),
            "coverage": round(kw, 3),
            "tokens": cnt,
            "budget_fit": round(fit, 3),
            "structure": round(st, 3),
            "repetition": round(rep, 3),
            "missing": kw_miss,
        })

    if require_all and keywords:
        passed = [s["index"] for s in scores if not s["missing"]]
        if passed:
            winner = max(passed, key=lambda i: (scores[i]["score"], -i))
            all_miss = False
        else:
            winner = max(range(n), key=lambda i: (scores[i]["coverage"],
                                                  scores[i]["score"], -i))
            all_miss = True
    else:
        winner = max(range(n), key=lambda i: (scores[i]["score"], -i))
        all_miss = False

    if require_all:
        return winner, scores, {"require_all": True, "all_miss": all_miss}
    return winner, scores, None
