"""End-to-end test of the ComfyUI node without launching ComfyUI."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from qwen35_prompt_enhancer import nodes


def main():
    print("models:", nodes._model_dirs())
    model = nodes._model_dirs()[0]
    node = nodes.Qwen35PromptEnhancer()

    for mode in ("keep_loaded", "offload_after_run"):
        t0 = time.perf_counter()
        res = node.enhance(
            text="a cat sitting on a windowsill",
            model=model,
            memory_mode=mode,
            max_tokens="512",
            system_prompt=nodes.DEFAULT_SYSTEM,
            temperature=0.7, top_p=0.9, top_k=40, seed=1234,
        )
        dt = time.perf_counter() - t0
        out = res["result"][0]
        print(f"\n[{mode}] {dt:.1f}s  {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM")
        print(repr(out))


if __name__ == "__main__":
    main()
