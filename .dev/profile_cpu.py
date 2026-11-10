"""Attribute CPU cost inside one decode step, by engine call site."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"


def main():
    import qwen35_prompt_enhancer.engine as E

    eng = E.Engine(MODEL_DIR, device="cuda", log=lambda *a: None)
    with torch.inference_mode():
        eng.prefill(list(range(1000, 1128)))
        for _ in range(5):
            eng.decode_step(1000)
    torch.cuda.synchronize()

    names = ["gemm_i8", "proj3", "mlp_gate_up", "rmsnorm_quant", "quant_row",
             "gate_proj", "conv_silu", "conv_state_update", "delta_decode",
             "norm_gated", "gate_mul_quant", "attn_decode", "qk_norm_rope"]
    stats = {n: [0.0, 0] for n in names}

    def wrap(fn, name):
        def inner(*a, **kw):
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            stats[name][0] += time.perf_counter() - t0
            stats[name][1] += 1
            return r
        return inner

    for n in names:
        setattr(E, n, wrap(getattr(E, n), n))

    # also count Workspace attribute accesses
    orig_getattr = E._WSView.__getattr__
    cnt = {"n": 0}

    def counting(self, name):
        cnt["n"] += 1
        return orig_getattr(self, name)

    E._WSView.__getattr__ = counting

    n = 20
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(n):
            eng.decode_step(1000)
        total = time.perf_counter() - t0
    torch.cuda.synchronize()

    print(f"\ntotal per token: {total/n*1e3:6.2f} ms")
    print(f"ws attr reads/token: {cnt['n']/n:.0f}")
    rows = sorted(((v[0] / n * 1e3, v[1] / n, k) for k, v in stats.items()), reverse=True)
    acc = 0.0
    for ms, calls, k in rows:
        acc += ms
        print(f"{ms:7.2f} ms  {calls:5.1f} calls  {ms/calls*1e3:6.1f} us/call  {k}")
    print(f"{acc:7.2f} ms  accounted")
    print(f"{(total/n*1e3)-acc:7.2f} ms  other (python plumbing)")


if __name__ == "__main__":
    main()
