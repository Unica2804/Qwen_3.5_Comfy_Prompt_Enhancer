"""Full-attention kernels (partial RoPE, GQA) and KV cache handling.

The KV cache has two encodings, selected by a compile-time `INT8_KV` flag:

* bf16 — `kvc` bf16 `[seq, 2*NKV*HD]` (the original, lossless path);
* int8 — `kvc` int8 `[seq, 2*NKV*HD]` plus `kvs` fp32 `[seq, 2*NKV]`, one
  scale per (position, head, side): the K vector of head `h` and the V vector
  of head `h` each carry their own `amax/127` scale.  Per-head granularity
  keeps the softmax-score precision close to bf16 while halving the cache size
  and the per-token KV read bandwidth at decode.  `Q` is never cached and
  stays bf16 either way.
"""
import torch
import triton
import triton.language as tl

from .common import round_i8


@triton.jit
def _norm_rope_store(src, w, dst, pos, invf, RD: tl.constexpr, HD: tl.constexpr,
                     eps, scale, ONE_PLUS: tl.constexpr,
                     kvs, kv_int8: tl.constexpr):
    """RMSNorm (1+w) over HD then partial rotate_half RoPE on the first RD dims.

    When `kv_int8` is set, writes int8 values to `dst` and the `amax/127`
    scale to `kvs` (already pointing at this head's scale slot).
    """
    D2: tl.constexpr = RD // 2
    d = tl.arange(0, D2)
    xlo = tl.load(src + d).to(tl.float32)
    xhi = tl.load(src + D2 + d).to(tl.float32)
    dr = tl.arange(0, HD)
    rmask = dr < (HD - RD)
    xr = tl.load(src + RD + dr, mask=rmask, other=0.0).to(tl.float32)

    wlo = tl.load(w + d).to(tl.float32)
    whi = tl.load(w + D2 + d).to(tl.float32)
    wr = tl.load(w + RD + dr, mask=rmask, other=0.0).to(tl.float32)
    if ONE_PLUS:
        wlo += 1.0
        whi += 1.0
        wr += 1.0

    var = (tl.sum(xlo * xlo) + tl.sum(xhi * xhi) + tl.sum(xr * xr)) / HD
    rstd = tl.rsqrt(var + eps)
    ylo = xlo * rstd * wlo
    yhi = xhi * rstd * whi
    yr = xr * rstd * wr

    theta = pos * tl.load(invf + d)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    plo = (ylo * cos - yhi * sin) * scale
    phi = (yhi * cos + ylo * sin) * scale
    if kv_int8:
        amax = tl.maximum(tl.max(tl.abs(plo)),
                          tl.maximum(tl.max(tl.abs(phi)), tl.max(tl.abs(yr * scale))))
        sc = tl.maximum(amax / 127.0, 1e-8)
        tl.store(dst + d, round_i8(plo / sc))
        tl.store(dst + D2 + d, round_i8(phi / sc))
        tl.store(dst + RD + dr, round_i8(yr * scale / sc), mask=rmask)
        tl.store(kvs, sc)
    else:
        tl.store(dst + d, plo.to(tl.bfloat16))
        tl.store(dst + D2 + d, phi.to(tl.bfloat16))
        tl.store(dst + RD + dr, (yr * scale).to(tl.bfloat16), mask=rmask)


@triton.jit
def qk_norm_rope_kernel(PROJ, QN, KN, INVF, QOUT, KVC, KVS, T, pos0,
                        NQ: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr,
                        RD: tl.constexpr, eps, stride_pm,
                        INT8_KV: tl.constexpr, BATCH: tl.constexpr,
                        KV_STRIDE: tl.constexpr, KVS_STRIDE: tl.constexpr):
    """grid = (T, NQ + 2*NKV): q heads, then k heads, then v heads."""
    row = tl.program_id(0)
    hi = tl.program_id(1)
    qscale = 1.0 / tl.sqrt(float(HD))
    pos = pos0 + row
    if BATCH:
        # rows are candidates: the position is shared, the cache slice is per row
        pos = pos0
    SNS: tl.constexpr = 2 * NKV  # scales per position (K and V heads)
    qb = KVC + row * KV_STRIDE
    sb = KVS + row * KVS_STRIDE

    if hi < NQ:
        _norm_rope_store(PROJ + row * stride_pm + hi * 2 * HD, QN,
                         QOUT + row * (NQ * HD) + hi * HD, pos, INVF,
                         RD, HD, eps, qscale, True, QOUT, False)
    elif hi < NQ + NKV:
        _norm_rope_store(PROJ + row * stride_pm + NQ * 2 * HD + (hi - NQ) * HD, KN,
                         qb + pos * SNS * HD + (hi - NQ) * HD, pos, INVF,
                         RD, HD, eps, 1.0, True,
                         sb + pos * SNS + (hi - NQ), INT8_KV)
    else:
        offs = tl.arange(0, HD)
        src = PROJ + row * stride_pm + NQ * 2 * HD + NKV * HD + (hi - NQ - NKV) * HD
        dst = qb + pos * SNS * HD + (NKV + (hi - NQ - NKV)) * HD
        if INT8_KV:
            v = tl.load(src + offs).to(tl.float32)
            sc = tl.maximum(tl.max(tl.abs(v)) / 127.0, 1e-8)
            tl.store(dst + offs, round_i8(v / sc))
            tl.store(sb + pos * SNS + NKV + (hi - NQ - NKV), sc)
        else:
            v = tl.load(src + offs)
            tl.store(dst + offs, v)


def qk_norm_rope(proj, qn, kn, invf, qout, kvc, kvs, pos0, eps=1e-6, kv_int8=False,
                 batch=False, kv_stride=None, kvs_stride=None):
    """Norm + RoPE the q/k/v heads; k/v go to the KV cache (bf16 or int8).

    In batch mode each row is one candidate at the *shared* position `pos0`;
    the cache is laid out as per-row slices of `kv_stride` elements
    (`kvs_stride` for the scales).
    """
    if kv_int8 and kvs is None:
        raise ValueError("kv_int8=True requires the kvs scales tensor")
    T = proj.shape[0]
    HD = qn.numel()
    NQ = qout.shape[1] // HD
    NKV = kvc.shape[1] // (2 * HD)
    RD = 2 * invf.numel()
    SNS = 2 * NKV
    qk_norm_rope_kernel[(T, NQ + 2 * NKV)](
        proj, qn, kn, invf, qout, kvc,
        kvs if kvs is not None else kvc,   # dummy when INT8_KV is False
        T, pos0, NQ=NQ, NKV=NKV, HD=HD, RD=RD, eps=eps, stride_pm=proj.stride(0),
        INT8_KV=kv_int8, BATCH=batch,
        KV_STRIDE=kv_stride or 0, KVS_STRIDE=kvs_stride or 0, num_warps=4,
    )


@triton.jit
def attn_prefill_kernel(Q, KVC, KVS, OUT, T, pos0, NQ: tl.constexpr,
                        NKV: tl.constexpr, HD: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        INT8_KV: tl.constexpr):
    pid = tl.program_id(0)
    head = pid % NQ
    mb = pid // NQ
    kvh = head // (NQ // NKV)
    SNS: tl.constexpr = 2 * NKV

    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + offs_m[:, None] * (NQ * HD) + head * HD + offs_d[None, :],
                mask=(offs_m < T)[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HD), tl.float32)

    end = tl.minimum((mb + 1) * BLOCK_M, T)
    # Keys live at absolute positions 0 .. pos0+len(chunk_b)-1: a chunked prefill
    # must attend to everything the earlier chunks wrote into the cache.
    total = pos0 + end
    for start_n in range(0, total, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < total
        if INT8_KV:
            k_raw = tl.load(KVC + offs_n[:, None] * (SNS * HD) + kvh * HD + offs_d[None, :],
                            mask=nmask[:, None], other=0)
            ks = tl.load(KVS + offs_n * SNS + kvh, mask=nmask, other=0.0)
            k = (k_raw.to(tl.float32) * ks[:, None]).to(tl.bfloat16)
        else:
            k = tl.load(KVC + offs_n[:, None] * (SNS * HD) + kvh * HD + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        causal = ((pos0 + offs_m)[:, None] >= offs_n[None, :]) & nmask[None, :]
        s = tl.where(causal, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if INT8_KV:
            v_raw = tl.load(KVC + offs_n[:, None] * (SNS * HD) + (NKV + kvh) * HD + offs_d[None, :],
                            mask=nmask[:, None], other=0)
            vs = tl.load(KVS + offs_n * SNS + NKV + kvh, mask=nmask, other=0.0)
            v = (v_raw.to(tl.float32) * vs[:, None]).to(tl.bfloat16)
        else:
            v = tl.load(KVC + offs_n[:, None] * (SNS * HD) + (NKV + kvh) * HD + offs_d[None, :],
                        mask=nmask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(OUT + offs_m[:, None] * (NQ * HD) + head * HD + offs_d[None, :],
             out.to(tl.bfloat16), mask=(offs_m < T)[:, None])


def attn_prefill(q, kvc, out, T, pos0=0, kvs=None, kv_int8=False, block_m=32,
                 block_n=32, nq=8, nkv=2, hd=256):
    """Causal prefill over the KV cache; reads the int8 cache when kv_int8."""
    if kv_int8 and kvs is None:
        raise ValueError("kv_int8=True requires the kvs scales tensor")
    grid = (triton.cdiv(T, block_m) * nq,)
    attn_prefill_kernel[grid](
        q, kvc, kvs if kvs is not None else kvc, out, T, pos0,
        NQ=nq, NKV=nkv, HD=hd, BLOCK_M=block_m, BLOCK_N=block_n,
        INT8_KV=kv_int8, num_warps=4, num_stages=2)
    return out


@triton.jit
def attn_decode_kernel(Q, KVC, KVS, OUT, pos, NQ: tl.constexpr, NKV: tl.constexpr,
                       HD: tl.constexpr, BLOCK_N: tl.constexpr,
                       INT8_KV: tl.constexpr, KV_STRIDE: tl.constexpr,
                       KVS_STRIDE: tl.constexpr):
    head = tl.program_id(0)
    b = tl.program_id(1)
    kvh = head // (NQ // NKV)
    SNS: tl.constexpr = 2 * NKV
    offs_d = tl.arange(0, HD)
    q = tl.load(Q + b * NQ * HD + head * HD + offs_d).to(tl.float32)
    qb = KVC + b * KV_STRIDE
    sb = KVS + b * KVS_STRIDE

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HD,), tl.float32)
    n = pos + 1
    for start_n in range(0, n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < n
        if INT8_KV:
            ks = tl.load(sb + offs_n * SNS + kvh, mask=nmask, other=0.0)
            k = tl.load(qb + offs_n[:, None] * (SNS * HD) + kvh * HD + offs_d[None, :],
                        mask=nmask[:, None], other=0).to(tl.float32) * ks[:, None]
        else:
            k = tl.load(qb + offs_n[:, None] * (SNS * HD) + kvh * HD + offs_d[None, :],
                        mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1)
        s = tl.where(nmask, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha
        if INT8_KV:
            vs = tl.load(sb + offs_n * SNS + NKV + kvh, mask=nmask, other=0.0)
            v = tl.load(qb + offs_n[:, None] * (SNS * HD) + (NKV + kvh) * HD + offs_d[None, :],
                        mask=nmask[:, None], other=0).to(tl.float32) * vs[:, None]
        else:
            v = tl.load(qb + offs_n[:, None] * (SNS * HD) + (NKV + kvh) * HD + offs_d[None, :],
                        mask=nmask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(OUT + b * NQ * HD + head * HD + offs_d, (acc / l_i).to(tl.bfloat16))


def attn_decode(q, kvc, out, pos, kvs=None, kv_int8=False, batch=1, block_n=32,
                nq=8, nkv=2, hd=256):
    """Single-query decode read of the KV cache (bf16 or int8), batched."""
    if kv_int8 and kvs is None:
        raise ValueError("kv_int8=True requires the kvs scales tensor")
    attn_decode_kernel[(nq, batch)](
        q, kvc, kvs if kvs is not None else kvc, out, pos,
        NQ=nq, NKV=nkv, HD=hd, BLOCK_N=block_n, INT8_KV=kv_int8,
        KV_STRIDE=0 if batch == 1 else kvc.numel() // batch,
        KVS_STRIDE=0 if (batch == 1 or kvs is None) else kvs.numel() // batch,
        num_warps=4)
    return out
