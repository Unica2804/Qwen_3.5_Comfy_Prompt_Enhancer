"""Compare the Triton engine against the dumped reference activations."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from qwen35_prompt_enhancer.engine import Engine

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"

ref = torch.load("ref_dump.pt", weights_only=False)
ids = ref["ids"]
eng = Engine(MODEL_DIR, device="cuda", log=print)
eng._collect = True
with torch.inference_mode():
    logits = eng.forward(ids)

def cmp(name, a, b):
    a = a.float().cpu()
    b = b.float().cpu()
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    r = (d.norm() / (b.norm() + 1e-9)).item()
    i = int(d.argmax())
    print(f"{name:12s} max_abs={d.max().item():9.5f} rel_l2={r:.5f} cos={cos:.6f} "
          f"@[{i}] mine={a[i].item():.3f} ref={b[i].item():.3f} |ref|max={b.abs().max().item():.2f}")

for i, h in enumerate(eng.debug):
    cmp(f"layer {i}", h, ref["hidden"][i + 1])
cmp("logits", logits, ref["logits"])
print("argmax mine/ref:", int(logits.argmax()), int(ref["logits"].argmax()))
top = torch.topk(logits.float().cpu(), 5)
print("mine top5", top.indices.tolist(), [round(v, 2) for v in top.values.tolist()])
top = torch.topk(ref["logits"], 5)
print("ref  top5", top.indices.tolist(), [round(v, 2) for v in top.values.tolist()])
