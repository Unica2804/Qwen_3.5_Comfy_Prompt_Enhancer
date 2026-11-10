"""Profile one decode step: per-kernel time attribution."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"


def main():
    from qwen35_prompt_enhancer.engine import Engine

    eng = Engine(MODEL_DIR, device="cuda", log=lambda *a: None)
    ids = list(range(1000, 1000 + 128))
    with torch.inference_mode():
        eng.prefill(ids)
        for _ in range(5):
            eng.decode_step(1000)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            for _ in range(10):
                eng.decode_step(1000)
        torch.cuda.synchronize()

    ev = prof.key_averages()
    rows = [(e.self_device_time_total, e.key) for e in ev if e.self_device_time_total > 0]
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows) / 10.0
    print(f"\ntotal GPU time per token: {total:.2f} ms")
    for t, k in rows[:22]:
        print(f"{t/10.0:8.3f} ms  {t/total*100:5.1f}%  {k[:70]}")


if __name__ == "__main__":
    main()
