"""Reference model built with HF transformers, dequantized from the W8A16 checkpoint.

Used only for numerical parity testing of the Triton engine.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"
CKPT = os.path.join(MODEL_DIR, "qwen_0.8b_w8a16.safetensors")


@torch.no_grad()
def load_reference(device="cuda", dtype=torch.bfloat16):
    cfg = Qwen3_5TextConfig.from_pretrained(MODEL_DIR)
    model = Qwen3_5ForCausalLM(cfg)
    model.eval()

    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        keys = list(f.keys())
        quant = {k[: -len(".qweight")] for k in keys if k.endswith(".qweight")}
        for k in keys:
            if k.endswith(".scale"):
                continue
            t = f.get_tensor(k)
            if k.endswith(".qweight"):
                base = k[: -len(".qweight")]
                scale = f.get_tensor(base + ".scale").to(torch.float32)
                t = (t.to(torch.float32).T.contiguous() * scale.unsqueeze(1)).to(dtype)
                sd[base + ".weight"] = t
            else:
                sd[k] = t.to(dtype) if t.is_floating_point() else t

    # tied embeddings: lm_head shares embed_tokens
    sd.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if "lm_head" not in m], missing
    assert not unexpected, unexpected
    model = model.to(device=device, dtype=dtype)
    return model


if __name__ == "__main__":
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    model = load_reference()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about the ocean."},
    ]
    tmpl = open(os.path.join(MODEL_DIR, "chat_template.jinja"), encoding="utf-8").read()
    from jinja2 import Template

    text = Template(tmpl).render(
        messages=msgs, add_generation_prompt=True, enable_thinking=False
    )
    ids = torch.tensor([tok.encode(text).ids], device="cuda")
    print("prompt tokens:", ids.shape)
    out = model(ids, use_cache=True)
    print("logits", out.logits.shape, out.logits.dtype)
    nxt = int(out.logits[0, -1].argmax())
    print("next token:", nxt, repr(tok.decode([nxt])))
    torch.save(out.logits[0, -8:].float().cpu(), "ref_logits.pt")
    # quick greedy decode with transformers
    gen = model.generate(
        ids,
        max_new_tokens=48,
        do_sample=False,
        eos_token_id=248046,
        pad_token_id=248044,
    )
    print(tok.decode(gen[0, ids.shape[1]:].tolist()))
