"""W4 GEMM kernels: int4 weights unpacked to int8 *inside* the dot loop.

The engine W4 layout is K-major so the GEMM reads it as coalesced runs along N:

* `W`  — `uint8 [K//2, N]`, even-K element in the low nibble, odd-K in the high
  nibble (byte j holds k = 2j and 2j+1);
* `WS` — `bf16/float [K//WGRP, N]`, one scale per `WGRP`-element group along K;
* `Z`  — optional `uint8 [K//WGRP, N]`, one *real* zero per group for
  asymmetric checkpoints (AWQ-style, and GPTQ with the `+1` baked at load).
  A stored element decodes to `signed = nibble - z`; the symmetric grid is the
  special case `z = 8` for every group.

Within one sub-block the nibbles are re-assembled into a contiguous int8 tile
`[SUB, N]` and fed to int8 tensor-core dots with int32 accumulation; scales
multiply in the epilogue only, after the exact integer part is complete.
`SUB = min(WGRP, 64)`:

* `SUB == BLOCK_K (64)` — native groups of 64+ (multiples of 64).  Fast path:
  the even/odd K split of the tile is free (`reshape` + `tl.split`), giving two
  dots of K=32 with no shuffle, and the whole block shares one scale row.
* `SUB == 32` — weight groups of 32: one dot per group (K=32 satisfies
  Triton's minimum), the even/odd nibble halves are re-interleaved with
  `tl.trans` + `tl.interleave`.  Slightly slower, used by compressed-tensors
  g32 checkpoints and requantized fine-grained GPTQ g16 checkpoints.

Group sizes below 32 cannot be represented (`tl.dot` needs K >= 32);
checkpoints on such grids are dequantized and requantized at load time (see
`quant_formats.py`).
"""
import triton
import triton.language as tl

from .common import GROUP, GROUP_C, round_i8, silu, swizzle

# int4 values live in [-8, 7] once the nibble is biased by 8.
# Triton kernels can only read globals declared as tl.constexpr.
W4_ZERO_C = tl.constexpr(8)


@triton.jit
def _w4_dot(a, lo, hi, BLOCK_M: tl.constexpr, SUB: tl.constexpr):
    """int32 sum over one sub-block: even-K half against `lo`, odd-K against `hi`."""
    SH: tl.constexpr = SUB // 2
    ae, ao = tl.split(tl.reshape(a, (BLOCK_M, SH, 2)))
    return tl.dot(ae, lo, out_dtype=tl.int32) + tl.dot(ao, hi, out_dtype=tl.int32)


@triton.jit
def _w4_tile(v, z, HAS_Z: tl.constexpr):
    """Packed bytes [SUB//2, N] -> int8 tile [SUB, N] in natural K order."""
    lo = v & 0x0F
    hi = (v >> 4) & 0x0F
    if HAS_Z:
        lo = lo - z
        hi = hi - z
    else:
        lo = lo - W4_ZERO_C
        hi = hi - W4_ZERO_C
    # interleave along K: (even k low nibble, odd k high nibble) -> k order
    return tl.trans(tl.interleave(tl.trans(lo.to(tl.int8)),
                                  tl.trans(hi.to(tl.int8))))


@triton.jit
def _w4_acc(A, W, WS, Z, om, cn, m_mask, n_mask, asc, K, SW, k0,
            BLOCK_M: tl.constexpr, BK: tl.constexpr, SUB: tl.constexpr,
            WGRP: tl.constexpr, HAS_Z: tl.constexpr):
    """One sub-block of the W4 GEMM: int8 dots + scale epilogue.

    `cn` holds the weight-column indices (offs_n for the GEMM, anything for
    proj3), `SW` the weight row stride, `asc` the preloaded activation scales
    for this 64-block.  Returns the fp32 contribution for this sub-block.
    """
    if SUB == BK:
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(A + om[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        jj = tl.arange(0, BK // 2)
        v = tl.load(W + (k0 // 2 + jj)[:, None] * SW + cn[None, :],
                    mask=n_mask[None, :], other=0).to(tl.int32)
        srow = k0 // WGRP
        if HAS_Z:
            z = tl.load(Z + srow * SW + cn[None, :], mask=n_mask[None, :], other=0)
            lo = ((v & 0x0F) - z).to(tl.int8)
            hi = (((v >> 4) & 0x0F) - z).to(tl.int8)
        else:
            lo = ((v & 0x0F) - W4_ZERO_C).to(tl.int8)
            hi = (((v >> 4) & 0x0F) - W4_ZERO_C).to(tl.int8)
        acc = _w4_dot(a, lo, hi, BLOCK_M, BK)
    else:
        offs_k = k0 + tl.arange(0, SUB)
        a = tl.load(A + om[:, None] * K + offs_k[None, :],
                    mask=m_mask[:, None] & (offs_k[None, :] < K), other=0)
        jj = tl.arange(0, SUB // 2)
        v = tl.load(W + (k0 // 2 + jj)[:, None] * SW + cn[None, :],
                    mask=n_mask[None, :], other=0).to(tl.int32)
        srow = k0 // WGRP
        if HAS_Z:
            z = tl.load(Z + srow * SW + cn[None, :], mask=n_mask[None, :], other=0)
        else:
            z = v   # unused dummy (HAS_Z is constexpr-folded away)
        acc = tl.dot(a, _w4_tile(v, z, HAS_Z), out_dtype=tl.int32)
    wsc = tl.load(WS + srow * SW + cn, mask=n_mask, other=0.0).to(tl.float32)
    return acc.to(tl.float32) * (asc[:, None] * wsc[None, :])


@triton.jit
def gemm_w4_kernel(A, AS, W, WS, Z, C, M, N, K, NG,
                   stride_cm, RES, stride_rm,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                   ADD_RES: tl.constexpr, WGRP: tl.constexpr, HAS_Z: tl.constexpr):
    SUB: tl.constexpr = BLOCK_K if WGRP >= BLOCK_K else WGRP
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kbi in range(0, tl.cdiv(K, BLOCK_K)):
        asc = tl.load(AS + offs_m * NG + kbi, mask=m_mask, other=0.0)
        for it in tl.static_range(BLOCK_K // SUB):
            out += _w4_acc(A, W, WS, Z, offs_m, offs_n, m_mask, n_mask,
                           asc, K, N, kbi * BLOCK_K + it * SUB,
                           BLOCK_M, BLOCK_K, SUB, WGRP, HAS_Z)

    if ADD_RES:
        out += tl.load(RES + offs_m[:, None] * stride_rm + offs_n[None, :],
                       mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def gemm_w4(a_q, a_s, b, b_s, out, residual=None, b_z=None, block_m=None,
            block_n=None, block_k=64, group_m=8, num_warps=None, num_stages=3):
    """out[M,N] (bf16) = (a_q * a_s) @ unpack(b) * b_s, with optional residual add.

    `b` is [K//2, N] uint8 packed nibbles, `b_s` is [K//WGRP, N] group scales,
    and `b_z` an optional [K//WGRP, N] uint8 per-group real zero (asymmetric
    checkpoints).  `WGRP` may be 32, 64, or any multiple of 64.
    """
    M, K = a_q.shape
    N = b.shape[1]
    if block_m is None:
        block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    if block_n is None:
        block_n = (64 if N <= 4096 else 256) if M <= 16 else 128
    if num_warps is None:
        num_warps = 4 if block_m <= 16 else 8
    wgrp = K // b_s.shape[0]
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    res = residual if residual is not None else out
    gemm_w4_kernel[grid](
        a_q, a_s, b, b_s, b_z if b_z is not None else b, out, M, N, K,
        a_s.shape[1], out.stride(0), res,
        residual.stride(0) if residual is not None else 0,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        ADD_RES=residual is not None, WGRP=wgrp, HAS_Z=b_z is not None,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


@triton.jit
def proj3_w4_kernel(A, AS, W1, S1, Z1, W2, S2, Z2, W3, S3, Z3, C,
                    M, N1, N2, N3, K, NG, stride_c,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                    WGRP: tl.constexpr, HAS_Z: tl.constexpr):
    """Three W4 projections sharing one activation tile (q/k/v, or qkv+z)."""
    SUB: tl.constexpr = BLOCK_K if WGRP >= BLOCK_K else WGRP
    pid = tl.program_id(0)
    NT = N1 + N2 + N3
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(NT, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M

    if pid_n * BLOCK_N < N1:
        W = W1
        WS = S1
        ZZ = Z1
        NOFF = 0
        NN = N1
    elif pid_n * BLOCK_N < N1 + N2:
        W = W2
        WS = S2
        ZZ = Z2
        NOFF = N1
        NN = N2
    else:
        W = W3
        WS = S3
        ZZ = Z3
        NOFF = N1 + N2
        NN = N3
    col = offs_n - NOFF
    n_mask = (col >= 0) & (col < NN)

    out = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kbi in range(0, tl.cdiv(K, BLOCK_K)):
        asc = tl.load(AS + offs_m * NG + kbi, mask=m_mask, other=0.0)
        for it in tl.static_range(BLOCK_K // SUB):
            out += _w4_acc(A, W, WS, ZZ, offs_m, col, m_mask, n_mask,
                           asc, K, NN, kbi * BLOCK_K + it * SUB,
                           BLOCK_M, BLOCK_K, SUB, WGRP, HAS_Z)

    tl.store(C + offs_m[:, None] * stride_c + offs_n[None, :], out.to(tl.bfloat16),
             mask=m_mask[:, None] & n_mask[None, :])


def proj3_w4(a_q, a_s, w1, s1, w2, s2, w3, s3, out, z1=None, z2=None, z3=None,
             block_k=64):
    """Three W4 projections sharing one activation tile; per-tensor group zeros."""
    M, K = a_q.shape
    n1 = w1.shape[1] if w1 is not None else 0
    n2 = w2.shape[1] if w2 is not None else 0
    n3 = w3.shape[1] if w3 is not None else 0
    if w2 is None:
        w2, s2 = w1, s1
    if w3 is None:
        w3, s3 = w1, s1
    block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    block_n = 64 if M <= 16 else 128
    wgrp = K // s1.shape[0]
    grid = (triton.cdiv(M, block_m) * triton.cdiv(n1 + n2 + n3, block_n),)
    if z1 is None and z2 is None and z3 is None:
        z1u = z2u = z3u = a_s          # unused (HAS_Z is False)
    else:
        fill = next(z for z in (z1, z2, z3) if z is not None)   # uint8 dummy
        z1u = z1 if z1 is not None else fill
        z2u = z2 if z2 is not None else fill
        z3u = z3 if z3 is not None else fill
    proj3_w4_kernel[grid](
        a_q, a_s, w1, s1, z1u,
        w2, s2, z2u,
        w3, s3, z3u,
        out, M, n1, n2, n3, K, a_s.shape[1], out.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        WGRP=wgrp, HAS_Z=z1 is not None or z2 is not None or z3 is not None,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out


@triton.jit
def mlp_gate_up_quant_w4_kernel(A, AS, WG, SG, ZG, WU, SU, ZU, Q, QS, M, N, K, NG,
                                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                                NG_B: tl.constexpr, WGRP: tl.constexpr,
                                HAS_Z: tl.constexpr):
    """h = silu(gate(x)) * up(x) with int4 weights (W4A8-dynamic), emitted as
    int8 + per-64 group scales for the subsequent int8 GEMM."""
    SUB: tl.constexpr = BLOCK_K if WGRP >= BLOCK_K else WGRP
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = swizzle(pid, num_pid_m, num_pid_n, GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    g = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    u = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for kbi in range(0, tl.cdiv(K, BLOCK_K)):
        asc = tl.load(AS + offs_m * NG + kbi, mask=m_mask, other=0.0)
        for it in tl.static_range(BLOCK_K // SUB):
            k0 = kbi * BLOCK_K + it * SUB
            g += _w4_acc(A, WG, SG, ZG, offs_m, offs_n, m_mask, n_mask,
                         asc, K, N, k0, BLOCK_M, BLOCK_K, SUB, WGRP, HAS_Z)
            u += _w4_acc(A, WU, SU, ZU, offs_m, offs_n, m_mask, n_mask,
                         asc, K, N, k0, BLOCK_M, BLOCK_K, SUB, WGRP, HAS_Z)

    h = silu(g) * u
    # Match the bf16 activation precision of the reference dataflow.
    h = h.to(tl.bfloat16).to(tl.float32)

    h3 = tl.reshape(h, (BLOCK_M, NG_B, GROUP_C))
    amax = tl.max(tl.abs(h3), axis=2)
    sc = tl.maximum(amax / 127.0, 1e-8)
    q3 = round_i8(h3 / sc[:, :, None])
    tl.store(Q + offs_m[:, None] * N + offs_n[None, :],
             tl.reshape(q3, (BLOCK_M, BLOCK_N)),
             mask=m_mask[:, None] & n_mask[None, :])
    og = pid_n * (BLOCK_N // GROUP_C) + tl.arange(0, NG_B)
    tl.store(QS + offs_m[:, None] * (N // GROUP_C) + og[None, :], sc,
             mask=m_mask[:, None])


def mlp_gate_up_quant_w4(a_q, a_s, wg, sg, wu, su, out_q, out_s, zg=None, zu=None,
                         block_k=64):
    """W4 gate/up fusion; `zg`/`zu` optional per-group zeros ([K//WGRP, N])."""
    M, K = a_q.shape
    N = wg.shape[1]
    block_m = 16 if M <= 16 else (32 if M <= 64 else 64)
    block_n = 64 if M <= 16 else 128
    wgrp = K // sg.shape[0]
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    if zg is None and zu is None:
        zgu = zuu = a_s                # unused (HAS_Z is False)
    else:
        fill = zg if zg is not None else zu    # uint8 dummy
        zgu = zg if zg is not None else fill
        zuu = zu if zu is not None else fill
    mlp_gate_up_quant_w4_kernel[grid](
        a_q, a_s, wg, sg, zgu,
        wu, su, zuu,
        out_q, out_s, M, N, K, a_s.shape[1],
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=8,
        NG_B=block_n // GROUP, WGRP=wgrp, HAS_Z=zg is not None or zu is not None,
        num_warps=4 if block_m <= 16 else 8, num_stages=3,
    )
    return out_q, out_s
