"""Greedy generation parity: Triton engine vs HF reference tokens."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"
OUT = "ref_gen.pt"


def ref_tokens():
    from tokenizers import Tokenizer
    from jinja2 import Template
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ref_model import load_reference

    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    tmpl = open(os.path.join(MODEL_DIR, "chat_template.jinja"), encoding="utf-8").read()
    text = Template(tmpl).render(
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "Write a short poem about the ocean."}],
        add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text).ids
    model = load_reference()
    with torch.no_grad():
        gen = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=64,
                             do_sample=False, eos_token_id=248046, pad_token_id=248044)
    out = gen[0, len(ids):].tolist()
    torch.save({"ids": ids, "out": out}, OUT)
    print("ref tokens:", out)
    print(repr(tok.decode(out)))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ref"
    if mode == "ref":
        ref_tokens()
    else:
        from tokenizers import Tokenizer
        from qwen35_prompt_enhancer.engine import Engine
        d = torch.load(OUT, weights_only=False)
        tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        eng = Engine(MODEL_DIR, device="cuda")
        with torch.inference_mode():
            out = eng.generate(d["ids"], max_new_tokens=64, temperature=0.0)
        print("mine:", out)
        ref = d["out"]
        n = 0
        for a, b in zip(out, ref):
            if a != b:
                break
            n += 1
        print(f"match {n}/{min(len(out), len(ref))} tokens")
        print(repr(tok.decode(out)))
        print("ref  :", repr(tok.decode(ref)))
