"""Timing: prefill and decode throughput of the Triton engine."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"


def main():
    from qwen35_prompt_enhancer.engine import Engine

    eng = Engine(MODEL_DIR, device="cuda")
    ids = list(range(1000, 1000 + 256))

    # warmup
    with torch.inference_mode():
        eng.prefill(ids)
        for _ in range(5):
            eng.decode_step(1000)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        eng.prefill(ids)
    torch.cuda.synchronize()
    t_pre = time.perf_counter() - t0
    print(f"prefill 256 tok: {t_pre*1e3:.1f} ms  ({256/t_pre:.0f} tok/s)")

    n = 64
    eng.reset()
    with torch.inference_mode():
        eng.prefill(ids)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            eng.decode_step(1000)
        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t0
    print(f"decode: {t_dec/n*1e3:.2f} ms/tok  ({n/t_dec:.0f} tok/s)")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
