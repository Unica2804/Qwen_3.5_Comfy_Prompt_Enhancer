"""How expensive is one Triton launch vs the engine's Python plumbing?"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import triton
import triton.language as tl


@triton.jit
def noop(X):
    pass


def main():
    x = torch.zeros(1, device="cuda")
    for _ in range(3):
        noop[(1,)](x)
    torch.cuda.synchronize()

    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        noop[(1,)](x)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    print(f"bare triton launch      : {(t1-t0)/n*1e6:6.1f} us")

    # a real kernel with a grid, to include arg processing
    from qwen35_prompt_enhancer.kernels.norm import quant_row_kernel

    src = torch.randn(1, 1024, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(1, 1024, dtype=torch.int8, device="cuda")
    s = torch.empty(1, 16, dtype=torch.float32, device="cuda")
    for _ in range(3):
        quant_row_kernel[(1,)](src, q, s, 1, 1024, src.stride(0), 16,
                               BLOCK_K=1024, NG=16, num_warps=4)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        quant_row_kernel[(1,)](src, q, s, 1, 1024, src.stride(0), 16,
                               BLOCK_K=1024, NG=16, num_warps=4)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    print(f"real kernel launch      : {(t1-t0)/n*1e6:6.1f} us")

    # workspace __getattr__ cost
    from qwen35_prompt_enhancer.engine import Workspace

    import json
    cfg = json.load(open(r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen\config.json", encoding="utf-8"))
    ws = Workspace(cfg, 512, torch.device("cuda"))
    v = ws.view(1)
    n2 = 20000
    t0 = time.perf_counter()
    for _ in range(n2):
        _ = v.hidden
        _ = v.xq
        _ = v.xs
        _ = v.proj
        _ = v.mixed
    t1 = time.perf_counter()
    print(f"ws.view attr (5 reads)  : {(t1-t0)/n2*1e6:6.1f} us")

    # dict weight lookup
    w = {"k": torch.zeros(1)}
    t0 = time.perf_counter()
    for _ in range(n2):
        _ = w["k"]
    t1 = time.perf_counter()
    print(f"dict lookup             : {(t1-t0)/n2*1e6:6.1f} us")


if __name__ == "__main__":
    main()
