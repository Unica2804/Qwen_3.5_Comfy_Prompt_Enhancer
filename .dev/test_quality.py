"""Teacher-forced quality check.

Runs the reference model over prompt + reference continuation, records logits at
every position, then does the same with the Triton engine and reports how often
the argmax agrees and how close the distributions are.  Unlike free-running
greedy comparison this is not chaotic, so it measures real fidelity.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

MODEL_DIR = r"D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\LLM\qwen"
CACHE = "tf_ref.pt"


def build_ref():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ref_model import load_reference

    d = torch.load("ref_gen.pt", weights_only=False)
    ids = d["ids"] + d["out"]
    model = load_reference()
    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda"), use_cache=False).logits[0]
    logits = logits.float().cpu()
    torch.save({"ids": ids, "logits": logits}, CACHE)
    print("ref logits", tuple(logits.shape))


def check():
    from qwen35_prompt_enhancer.engine import Engine

    d = torch.load(CACHE, weights_only=False)
    ids, ref = d["ids"], d["logits"]
    eng = Engine(MODEL_DIR, device="cuda")
    eng.reset()
    with torch.inference_mode():
        eng.forward(ids, pos0=0)
        mine = eng.logits_all(len(ids)).float().cpu()

    # positions where a token is actually predicted: 0..n-2
    m, r = mine[:-1], ref[:-1]
    agree = (m.argmax(-1) == r.argmax(-1)).float().mean().item()
    cos = torch.nn.functional.cosine_similarity(m, r, dim=-1)
    ref_lp = torch.log_softmax(r, dim=-1)
    mine_lp = torch.log_softmax(m, dim=-1)
    kl = (ref_lp.exp() * (ref_lp - mine_lp)).sum(-1)
    print(f"top-1 agreement : {agree*100:.1f}%")
    print(f"logit cosine    : mean {cos.mean():.5f}  min {cos.min():.5f}")
    print(f"KL(ref||mine)   : mean {kl.mean():.4f}  max {kl.max():.4f}")
    top5 = (r.topk(5, -1).indices.unsqueeze(-1)
            == m.topk(5, -1).indices.unsqueeze(1)).any(-1)
    print(f"ref top-1 in mine top-5: {top5[:, 0].float().mean().item()*100:.1f}%")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    build_ref() if mode == "ref" else check()
