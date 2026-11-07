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


# ---------------------------------------------------------------- best-of-N

_STOPWORDS = set(
    """a an the and or of in on at for with to into from by is are was were
    be it its this that these those as if so too very can all any one two
    """.split())


def _stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def keywords_of(text):
    """Unique stemmed content words of a user idea, for the coverage score."""
    words = re.findall(r"[a-z]{4,}", text.lower())
    seen = set()
    out = []
    for w in words:
        s = _stem(w)
        if w not in _STOPWORDS and s not in seen:
            seen.add(s)
            out.append(w)
    return out


def score_candidates(texts, keywords, budget=None, headers=None, tok=None):
    """Deterministic candidate scores (2.2).

    keyword coverage 40 % + budget fit 25 % + structure 20 % + repetition
    15 %.  Returns (winner_index, per-candidate score dicts).
    """
    scores = []
    n = len(texts)
    for idx, text in enumerate(texts):
        low = re.sub(r"[^a-z]+", " ", text.lower())
        stems = " ".join(_stem(w) for w in low.split())
        kw = 1.0
        if keywords:
            kw = sum(1 for k in keywords if _stem(k) in stems) / len(keywords)
        cnt = len(tok.encode(text)) if tok is not None else len(text.split())
        fit = 1.0 if not budget else min(1.0, budget / max(1, cnt))
        st = 1.0
        if headers:
            pos = [text.lower().find(h.lower() + ":") for h in headers]
            st = 1.0 if all(p >= 0 for p in pos) and pos == sorted(pos) else 0.0
        grams = re.findall(r"\S+ \S+ \S+ \S+", low)
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
        })
    winner = max(range(n), key=lambda i: (scores[i]["score"], -i))
    return winner, scores
