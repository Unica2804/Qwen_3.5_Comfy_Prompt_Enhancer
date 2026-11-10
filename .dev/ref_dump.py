"""Dump reference (bf16 HF) hidden states + logits for parity checking."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from ref_model import load_reference
from tokenizers import Tokenizer
from jinja2 import Template

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
tmpl = open(os.path.join(MODEL_DIR, "chat_template.jinja"), encoding="utf-8").read()
text = Template(tmpl).render(
    messages=[{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "Write a short poem about the ocean."}],
    add_generation_prompt=True, enable_thinking=False)
ids = tok.encode(text).ids

model = load_reference()
with torch.no_grad():
    out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False)
hs = [h[0, -1].float().cpu() for h in out.hidden_states]
logits = out.logits[0, -1].float().cpu()
torch.save({"ids": ids, "hidden": hs, "logits": logits}, "ref_dump.pt")
print("dumped", len(hs), "hidden states, logits", logits.shape)
print("argmax", int(logits.argmax()))
