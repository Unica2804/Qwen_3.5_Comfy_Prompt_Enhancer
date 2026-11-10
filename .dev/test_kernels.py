"""Numerical parity tests: Triton kernels vs. torch reference."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from qwen35_prompt_enhancer.kernels.attention import attn_decode, attn_prefill, qk_norm_rope
from qwen35_prompt_enhancer.kernels.gemm import gemm_i8, mlp_gate_up, proj3
from qwen35_prompt_enhancer.kernels.linear_attn import conv_silu, conv_state_update, delta_decode, delta_prefill
from qwen35_prompt_enhancer.kernels.norm import (gate_mul_quant, gate_proj, norm_gated, quant_row,
                                                rmsnorm_quant, silu_mul_quant)

torch.manual_seed(0)
dev = "cuda"


def rel(a, b):
    a = a.float()
    b = b.float()
    return (a - b).abs().max().item(), (a - b).norm().item() / (b.norm().item() + 1e-9)


def report(name, a, b):
    m, r = rel(a, b)
    print(f"{name:24s} max_abs={m:.5f} rel_l2={r:.5f}")


def test_gemm():
    G = 64
    for (M, N, K) in [(1, 3584, 1024), (37, 3584, 1024), (37, 2048, 1024), (1, 248320, 1024)]:
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
        b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
        as_ = torch.rand(M, K // G, device=dev) * 0.02
        bs = torch.rand(N, device=dev) * 0.002
        out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        gemm_i8(a, as_, b, bs, out, block_k=64)
        af = a.float() * as_.repeat_interleave(G, dim=1)
        ref = (af @ b.float()) * bs[None, :]
        report(f"gemm {M}x{N}x{K}", out, ref)

        res = torch.randn((M, N), dtype=torch.bfloat16, device=dev)
        gemm_i8(a, as_, b, bs, out, residual=res, block_k=64)
        report(f"gemm+res {M}x{N}x{K}", out, ref + res.float())


def test_proj3():
    G, M, K = 64, 37, 1024
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    as_ = torch.rand(M, K // G, device=dev) * 0.02
    w1 = torch.randint(-127, 127, (K, 6144), dtype=torch.int8, device=dev)
    w2 = torch.randint(-127, 127, (K, 2048), dtype=torch.int8, device=dev)
    s1 = torch.rand(6144, device=dev) * 0.002
    s2 = torch.rand(2048, device=dev) * 0.002
    out = torch.empty((M, 8192), dtype=torch.bfloat16, device=dev)
    proj3(a, as_, w1, s1, w2, s2, None, None, out)
    af = a.float() * as_.repeat_interleave(G, dim=1)
    ref1 = (af @ w1.float()) * s1[None, :]
    ref2 = (af @ w2.float()) * s2[None, :]
    report("proj3 part1", out[:, :6144], ref1)
    report("proj3 part2", out[:, 6144:], ref2)


def test_mlp():
    G, M, K, N = 64, 37, 1024, 3584
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    as_ = torch.rand(M, K // G, device=dev) * 0.02
    wg = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
    wu = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
    sg = torch.rand(N, device=dev) * 0.002
    su = torch.rand(N, device=dev) * 0.002
    out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    mlp_gate_up(a, as_, wg, sg, wu, su, out)
    af = a.float() * as_.repeat_interleave(G, dim=1)
    g = (af @ wg.float()) * sg[None, :]
    u = (af @ wu.float()) * su[None, :]
    ref = F.silu(g) * u
    report("mlp_gate_up", out, ref)


def test_norm():
    G, M, K = 64, 37, 1024
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w = torch.randn(K, dtype=torch.bfloat16, device=dev) * 0.2
    q = torch.empty(M, K, dtype=torch.int8, device=dev)
    s = torch.empty(M, K // G, dtype=torch.float32, device=dev)
    rmsnorm_quant(x, w, q, s)

    def group_q(y):
        yg = y.reshape(M, K // G, G)
        sc = yg.abs().amax(-1) / 127.0
        return torch.clamp(torch.round(yg / sc[:, :, None]), -127, 127).reshape(M, K)

    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * (w.float() + 1)
    report("rmsnorm_quant", q.float(), group_q(y))

    q2 = torch.empty(M, K, dtype=torch.int8, device=dev)
    s2 = torch.empty(M, K // G, device=dev)
    quant_row(x, q2, s2)
    report("quant_row", q2.float(), group_q(xf))


def test_gate_proj():
    G, M, K, NB = 64, 13, 1024, 16
    a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
    as_ = torch.rand(M, K // G, device=dev) * 0.02
    wb = torch.randn(NB, K, dtype=torch.bfloat16, device=dev) * 0.05
    wa = torch.randn(NB, K, dtype=torch.bfloat16, device=dev) * 0.05
    dt = torch.randn(NB, dtype=torch.bfloat16, device=dev)
    al = torch.randn(NB, dtype=torch.bfloat16, device=dev)
    beta = torch.empty(M, NB, device=dev)
    gate = torch.empty(M, NB, device=dev)
    gate_proj(a, as_, wb, wa, beta, gate, dt, al)
    x = a.float() * as_.repeat_interleave(G, dim=1)
    b = x @ wb.float().T
    av = x @ wa.float().T
    ref_beta = torch.sigmoid(b)
    ref_g = -torch.exp(al.float()) * F.softplus(av + dt.float())
    report("gate_proj beta", beta, ref_beta)
    report("gate_proj g", gate, ref_g)


def ref_delta(qkv, g, beta, state, T, NH=16, DK=128, DV=128, chunk=32):
    """Straight port of transformers' torch_chunk_gated_delta_rule."""
    q = qkv[:, : NH * DK].reshape(T, NH, DK)
    k = qkv[:, NH * DK: 2 * NH * DK].reshape(T, NH, DK)
    v = qkv[:, 2 * NH * DK:].reshape(T, NH, DV)
    out = torch.zeros(T, NH, DV, device=qkv.device, dtype=torch.float32)
    S = state.clone()
    for h in range(NH):
        qq = q[:, h].float()
        kk = k[:, h].float()
        vv = v[:, h].float()
        qq = qq * torch.rsqrt((qq * qq).sum(-1, keepdim=True) + 1e-6)
        kk = kk * torch.rsqrt((kk * kk).sum(-1, keepdim=True) + 1e-6)
        qq = qq * (1.0 / DK ** 0.5)
        sh = S[h].clone()
        for c0 in range(0, T, chunk):
            sl = slice(c0, min(T, c0 + chunk))
            qc, kc, vc = qq[sl], kk[sl], vv[sl]
            bc, gc = beta[sl, h].float(), g[sl, h].float()
            L = qc.shape[0]
            pad = chunk - L
            if pad:
                qc = F.pad(qc, (0, 0, 0, pad))
                kc = F.pad(kc, (0, 0, 0, pad))
                vc = F.pad(vc, (0, 0, 0, pad))
                bc = F.pad(bc, (0, pad))
                gc = F.pad(gc, (0, pad))
            kb = kc * bc[:, None]
            vb = vc * bc[:, None]
            gcs = gc.cumsum(0)
            decay = torch.tril(torch.exp(gcs[:, None] - gcs[None, :]))
            A = -(kb @ kc.T) * decay
            A = torch.tril(A, -1)
            Tm = torch.linalg.solve_triangular(torch.eye(chunk, device=qkv.device) - A,
                                               torch.eye(chunk, device=qkv.device), upper=False)
            value = Tm @ vb
            kcum = Tm @ (kb * gcs.exp()[:, None])
            v_new = value - kcum @ sh
            o = (qc * gcs.exp()[:, None]) @ sh + ((qc @ kc.T) * decay) @ v_new
            sh = sh * gcs[-1].exp() + (kc * (gcs[-1] - gcs).exp()[:, None]).T @ v_new
            out[c0: c0 + L, h] = o[:L]
        S[h] = sh
    return out.reshape(T, NH * DV), S


def test_delta():
    NH, DK, DV = 16, 128, 128
    C = 2 * NH * DK + NH * DV
    for T in (1, 5, 64, 70, 130):
        qkv = (torch.randn(T, C, device=dev) * 0.3).to(torch.bfloat16)
        g = -torch.rand(T, NH, device=dev) * 0.4
        beta = torch.rand(T, NH, device=dev)
        state = torch.randn(NH, DK, DV, device=dev) * 0.05
        out = torch.empty(T, NH * DV, dtype=torch.bfloat16, device=dev)
        st = state.clone()
        delta_prefill(qkv, g, beta, st, out, T)
        ref, ref_s = ref_delta(qkv, g, beta, state.clone(), T)
        report(f"delta_prefill T={T}", out, ref)
        report(f"delta_state  T={T}", st, ref_s)

    # decode path (single token)
    qkv = (torch.randn(1, C, device=dev) * 0.3).to(torch.bfloat16)
    g = -torch.rand(1, NH, device=dev) * 0.4
    beta = torch.rand(1, NH, device=dev)
    state = torch.randn(NH, DK, DV, device=dev) * 0.05
    out = torch.empty(1, NH * DV, dtype=torch.bfloat16, device=dev)
    st = state.clone()
    delta_decode(qkv[0], g[0], beta[0], st, out[0])
    ref, ref_s = ref_delta(qkv, g, beta, state.clone(), 1)
    report("delta_decode", out, ref)
    report("delta_decode_state", st, ref_s)


def test_conv():
    C = 2 * 16 * 128 + 16 * 128
    T = 23
    x = torch.randn(T, C, dtype=torch.bfloat16, device=dev)
    w = torch.randn(C, 4, dtype=torch.bfloat16, device=dev) * 0.1
    state = torch.randn(3, C, dtype=torch.bfloat16, device=dev)
    y = torch.empty(T, C, dtype=torch.bfloat16, device=dev)
    conv_silu(x, w, state, y, T)
    xf = x.float()
    st = state.float()
    full = torch.cat([st, xf], 0)
    out = torch.zeros(T, C, device=dev)
    for j in range(4):
        out += full[j: j + T] * w[:, j].float()[None, :]
    report("conv_silu", y, F.silu(out))
    st2 = state.clone()
    conv_state_update(x, st2, T)
    report("conv_state", st2, x[-3:].float())


def test_attention():
    NQ, NKV, HD, T = 8, 2, 256, 70
    q = (torch.randn(T, NQ * HD, device=dev) * 0.2).to(torch.bfloat16)
    kv = (torch.randn(T, 2 * NKV * HD, device=dev) * 0.5).to(torch.bfloat16)
    out = torch.empty(T, NQ * HD, dtype=torch.bfloat16, device=dev)
    attn_prefill(q, kv, out, T)
    # reference
    qh = q.float().reshape(T, NQ, HD).transpose(0, 1)
    kh = kv.float()[:, : NKV * HD].reshape(T, NKV, HD).transpose(0, 1)
    vh = kv.float()[:, NKV * HD:].reshape(T, NKV, HD).transpose(0, 1)
    kh = kh.repeat_interleave(NQ // NKV, dim=0)
    vh = vh.repeat_interleave(NQ // NKV, dim=0)
    s = qh @ kh.transpose(1, 2)
    mask = torch.tril(torch.ones(T, T, device=dev, dtype=torch.bool))
    s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, -1)
    ref = (p @ vh).transpose(0, 1).reshape(T, NQ * HD)
    report("attn_prefill", out, ref)

    # decode at the last position
    out1 = torch.empty(1, NQ * HD, dtype=torch.bfloat16, device=dev)
    attn_decode(q[T - 1], kv, out1[0], T - 1)
    report("attn_decode", out1, ref[T - 1: T])


def test_qk_rope():
    T, NQ, NKV, HD = 9, 8, 2, 256
    proj = (torch.randn(T, NQ * HD * 2 + 2 * NKV * HD, device=dev) * 0.5).to(torch.bfloat16)
    qn = torch.randn(HD, dtype=torch.bfloat16, device=dev) * 0.2
    kn = torch.randn(HD, dtype=torch.bfloat16, device=dev) * 0.2
    invf = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64))
    qout = torch.empty(T, NQ * HD, dtype=torch.bfloat16, device=dev)
    kvc = torch.zeros(64, 2 * NKV * HD, dtype=torch.bfloat16, device=dev)
    qk_norm_rope(proj, qn, kn, invf, qout, kvc, 0, 1e-6)

    p = proj.float()
    qraw = p[:, : NQ * HD * 2].reshape(T, NQ, 2, HD)[:, :, 0]
    qgate = p[:, : NQ * HD * 2].reshape(T, NQ, 2, HD)[:, :, 1]
    kraw = p[:, NQ * HD * 2: NQ * HD * 2 + NKV * HD].reshape(T, NKV, HD)
    vraw = p[:, NQ * HD * 2 + NKV * HD:].reshape(T, NKV, HD)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (w.float() + 1)

    def rope(x, pos):
        d = 64
        freqs = torch.arange(0, d, 2, device=dev).float() / d
        inv = 1.0 / (1e7 ** freqs)
        th = pos[:, None].float() * inv[None, :]
        emb = torch.cat([th, th], -1)
        cos, sin = emb.cos(), emb.sin()
        x1, x2 = x[..., :32], x[..., 32:64]
        rot = torch.cat([-x2, x1], -1)
        cos = cos[:, None, :]; sin = sin[:, None, :]
        return torch.cat([x[..., :64] * cos + rot * sin, x[..., 64:]], -1)

    pos = torch.arange(T, device=dev)
    qr = rope(rms(qraw, qn), pos) / HD ** 0.5
    kr = rope(rms(kraw, kn), pos)
    report("qk q", qout, qr.reshape(T, NQ * HD))
    report("qk k", kvc[:T, :NKV * HD], kr.reshape(T, NKV * HD))
    report("qk v", kvc[:T, NKV * HD:], vraw.reshape(T, NKV * HD))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = {
        "gemm": test_gemm, "proj3": test_proj3, "mlp": test_mlp, "norm": test_norm,
        "gate": test_gate_proj, "delta": test_delta, "conv": test_conv,
        "attn": test_attention, "qk": test_qk_rope,
    }
    for k, fn in tests.items():
        if which in ("all", k):
            print(f"--- {k}")
            try:
                fn()
            except Exception as e:
                import traceback
                traceback.print_exc()
