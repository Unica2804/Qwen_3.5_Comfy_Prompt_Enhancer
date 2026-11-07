"""Sampling helpers.

Greedy / temperature / top-k / top-p / min-p, plus repetition and presence
penalties on tokens already generated — matching the native sampler's knobs
(`repetition_penalty=1.05`, `min_p=0.05`, `presence_penalty=0.0` defaults live on
the node; the engine itself stays neutral).

Warp order: penalties on the raw logits -> temperature -> top-k -> min-p ->
top-p -> softmax draw.
"""
import torch


@torch.no_grad()
def sample_logits(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None,
                  min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                  generated=None, mask=None):
    """logits: [vocab] tensor on GPU. Returns an int token id.

    `mask`, when given, is a `[vocab]` boolean tensor of allowed tokens;
    disallowed positions are set to `-inf` before penalties/temperature, so
    greedy, top-k and top-p all respect it (2.3).  `generated` is the list of
    token ids produced so far in this sequence; repetition/presence penalties
    are applied to every id that appears in it.
    """
    x = logits.float()
    if x is logits:
        x = x.clone()

    if mask is not None:
        x = x.masked_fill(~mask.to(x.device), float("-inf"))

    if generated and (repetition_penalty != 1.0 or presence_penalty != 0.0):
        prev = torch.unique(torch.tensor(generated, dtype=torch.long,
                                         device=x.device))
        vals = x[prev]
        if repetition_penalty != 1.0:
            vals = torch.where(vals > 0, vals / repetition_penalty,
                               vals * repetition_penalty)
        if presence_penalty != 0.0:
            vals = vals - presence_penalty
        x[prev] = vals

    if temperature is None or temperature <= 1e-5:
        return int(torch.argmax(x))
    x = x / float(temperature)
    if top_k and top_k > 0:
        k = min(int(top_k), x.numel())
        kth = torch.topk(x, k).values[-1]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    probs = torch.softmax(x, dim=-1)
    if min_p is not None and min_p > 0.0:
        keep = probs >= float(min_p) * probs.max()
        probs = torch.where(keep, probs, torch.zeros_like(probs))
    if top_p is not None and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cs = torch.cumsum(sp, dim=-1)
        drop = (cs - sp) > top_p
        sp = torch.where(drop, torch.zeros_like(sp), sp)
        probs = torch.zeros_like(probs).scatter(0, si, sp)
    if probs.sum() <= 0:
        return int(torch.argmax(x))
    return int(torch.multinomial(probs, 1, generator=generator))


def sample_batch(logits, temperature=0.0, top_k=0, top_p=1.0, seed=0, step=0,
                 min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                 out=None, alive=None, header_mask=None, line_start=None,
                 grammar=None, gstates=None):
    """Sample one token per row of a [B, V] logits tensor (2.2).

    Candidate b starts from the same base `seed` via `seed + 1 + b` (and a step
    offset), so a batch is fully reproducible; dead rows draw token 0 (ignored
    by the caller).  Returns a list of B int token ids.
    """
    B = logits.shape[0]
    if alive is None:
        alive = [True] * B
    if out is None:
        out = [[] for _ in range(B)]
    if line_start is None:
        line_start = [True] * B
    if grammar is not None and gstates is None:
        gstates = [grammar.start_state() for _ in range(B)]
    dev = logits.device
    toks = []
    for b in range(B):
        if not alive[b]:
            toks.append(0)
            continue
        if grammar is not None:
            m = grammar.mask(gstates[b])
        elif header_mask is not None and line_start[b]:
            m = header_mask
        else:
            m = None
        gen = None
        if temperature and seed is not None:
            gen = torch.Generator(device=dev).manual_seed(
                int(seed) * 1000003 + (b + 1) * 7919 + step)
        tok = sample_logits(logits[b], temperature, top_k, top_p, gen,
                            min_p=min_p, repetition_penalty=repetition_penalty,
                            presence_penalty=presence_penalty,
                            generated=out[b], mask=m)
        toks.append(int(tok))
        if grammar is not None:
            gstates[b] = grammar.advance(gstates[b], int(tok))
    return toks
