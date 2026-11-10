"""Is decode CPU-dispatch bound or GPU bound?"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"


def main():
    from qwen35_prompt_enhancer.engine import Engine

    eng = Engine(MODEL_DIR, device="cuda", log=lambda *a: None)
    with torch.inference_mode():
        eng.prefill(list(range(1000, 1128)))
        for _ in range(5):
            eng.decode_step(1000)
    torch.cuda.synchronize()

    n = 50
    with torch.inference_mode():
        # CPU time to enqueue all launches (no sync in between)
        t0 = time.perf_counter()
        for _ in range(n):
            eng.decode_step(1000)
        t_cpu = time.perf_counter() - t0
        torch.cuda.synchronize()
        t_all = time.perf_counter() - t0

    print(f"CPU enqueue per token : {t_cpu/n*1e3:6.2f} ms")
    print(f"CPU+GPU per token     : {t_all/n*1e3:6.2f} ms")
    print(f"GPU tail (queued work): {(t_all-t_cpu)/n*1e3:6.2f} ms")

    # count launches
    import triton
    cnt = {"n": 0}
    orig = triton.runtime.jit.JITFunction.run

    def counting(*a, **kw):
        cnt["n"] += 1
        return orig(*a, **kw)

    triton.runtime.jit.JITFunction.run = counting
    try:
        with torch.inference_mode():
            eng.decode_step(1000)
    finally:
        triton.runtime.jit.JITFunction.run = orig
    print(f"triton launches/token : {cnt['n']}")


if __name__ == "__main__":
    main()
