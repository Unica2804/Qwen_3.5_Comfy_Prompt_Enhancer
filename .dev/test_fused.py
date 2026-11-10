"""Verify the fused quantising kernels match the unfused composition."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from qwen35_prompt_enhancer.kernels.gemm import mlp_gate_up, mlp_gate_up_quant
from qwen35_prompt_enhancer.kernels.norm import norm_gated, norm_gated_quant, quant_row

dev = "cuda"
G = 64
torch.manual_seed(0)


def deq(q, s):
    return q.float() * s.repeat_interleave(G, dim=1)


def main():
    # --- MLP: fused int8 output vs (bf16 h -> quant_row)
    for M in (1, 16, 96):
        K, N = 1024, 3584
        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
        as_ = torch.rand(M, K // G, device=dev) * 0.02
        wg = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
        wu = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=dev)
        sg = torch.rand(N, device=dev) * 0.002
        su = torch.rand(N, device=dev) * 0.002

        h = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        mlp_gate_up(a, as_, wg, sg, wu, su, h)
        q0 = torch.empty((M, N), dtype=torch.int8, device=dev)
        s0 = torch.empty((M, N // G), dtype=torch.float32, device=dev)
        quant_row(h, q0, s0)

        q1 = torch.empty((M, N), dtype=torch.int8, device=dev)
        s1 = torch.empty((M, N // G), dtype=torch.float32, device=dev)
        mlp_gate_up_quant(a, as_, wg, sg, wu, su, q1, s1)

        exact = (q1 == q0).float().mean().item()
        d = (deq(q1, s1) - deq(q0, s0)).abs()
        print(f"mlp  M={M:3d}  int8 equal {exact*100:6.2f}%  "
              f"dequant max_abs {d.max().item():.6f}  rel {(d.norm()/(deq(q0,s0).norm()+1e-9)).item():.2e}")

    # --- norm_gated: fused vs bf16 -> quant_row
    for T in (1, 33):
        HD, NV = 128, 16
        N = NV * HD
        v = (torch.randn(T, N, device=dev) * 0.5).to(torch.bfloat16)
        z = (torch.randn(T, N, device=dev) * 0.5).to(torch.bfloat16)
        w = torch.randn(HD, dtype=torch.bfloat16, device=dev) * 0.2

        y = torch.empty((T, N), dtype=torch.bfloat16, device=dev)
        norm_gated(v, z, w, y)
        q0 = torch.empty((T, N), dtype=torch.int8, device=dev)
        s0 = torch.empty((T, N // G), dtype=torch.float32, device=dev)
        quant_row(y, q0, s0)

        q1 = torch.empty((T, N), dtype=torch.int8, device=dev)
        s1 = torch.empty((T, N // G), dtype=torch.float32, device=dev)
        norm_gated_quant(v, z, w, q1, s1)

        exact = (q1 == q0).float().mean().item()
        d = (deq(q1, s1) - deq(q0, s0)).abs()
        print(f"ng   T={T:3d}  int8 equal {exact*100:6.2f}%  "
              f"dequant max_abs {d.max().item():.6f}  rel {(d.norm()/(deq(q0,s0).norm()+1e-9)).item():.2e}")


if __name__ == "__main__":
    main()
