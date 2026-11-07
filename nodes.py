"""ComfyUI node: expand a short idea into a detailed prompt with a local
Qwen3.5 model running on a fused INT8 Triton kernel engine."""
import json
import os
import threading

import torch

try:
    import comfy.model_management as mm
    import comfy.utils
except Exception:  # running outside ComfyUI (tests)
    mm = None
    comfy = None

from .engine import Engine
from .targets import (SectionGrammar, TARGETS, build_header_mask, get_target,
                      keywords_of, postprocess, score_candidates)
from .tokenizer import DEFAULT_SYSTEM, ChatTokenizer
from .weights import discover_models

_HERE = os.path.dirname(os.path.abspath(__file__))
# package -> custom_nodes -> ComfyUI
_COMFY_ROOT = os.path.dirname(os.path.dirname(_HERE))
LLM_ROOT = os.path.join(_COMFY_ROOT, "models", "LLM")
try:  # prefer ComfyUI's own resolution when available
    import folder_paths

    LLM_ROOT = os.path.join(folder_paths.models_dir, "LLM")
except Exception:
    pass

# Additional folders scanned for Qwen3.5 checkpoints, on top of models/LLM.
# Override or extend with QWEN35_MODEL_PATHS (os.pathsep separated).
EXTRA_MODEL_ROOTS = [r"D:\models"]

MAX_TOKEN_CHOICES = ["512", "1024", "2048", "4096"]

# Head-room added on top of the engine estimate when asking the model manager
# to free memory: CUDA context, cuBLAS/Triton handles and allocator churn.
_VRAM_MARGIN = 512 * 1024 * 1024

_cache = {}
_tok_cache = {}
_lock = threading.Lock()


def _model_roots():
    roots = [LLM_ROOT]
    for p in os.environ.get("QWEN35_MODEL_PATHS", "").split(os.pathsep):
        p = p.strip().strip('"')
        if p:
            roots.append(p)
    roots.extend(EXTRA_MODEL_ROOTS)
    return roots


def _model_map():
    """label -> checkpoint directory, for every discoverable model."""
    return dict(discover_models(_model_roots()))


def _model_dirs():
    return list(_model_map())


def _resolve_model(name):
    path = _model_map().get(name)
    if path is None:
        if os.path.isdir(name):
            return name
        raise FileNotFoundError(f"model not found: {name}")
    return path


def _estimate_engine_vram(model_dir, kv_seq=2048):
    """Rough VRAM (bytes) one engine will occupy once loaded.

    Mirrors the encodings `weights.py` produces (int8 embed, W4 head, packed
    int4/int8 stacked projections) plus the runtime state, so eviction decisions
    can be based on what this engine actually needs instead of a fixed 2 GB.
    """
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    H = cfg["hidden_size"]
    V = cfg["vocab_size"]
    I = cfg["intermediate_size"]
    NL = cfg["num_hidden_layers"]
    layer_types = cfg["layer_types"]
    n_lin = sum(1 for x in layer_types if x == "linear_attention")
    n_full = NL - n_lin
    NH = cfg["linear_num_value_heads"]
    DK = cfg["linear_key_head_dim"]
    DV = cfg["linear_value_head_dim"]
    KD = cfg.get("linear_key_dim", NH * DK)
    conv_dim = 2 * KD + DV * NH
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    conv_k = cfg.get("linear_conv_kernel_dim", 4)

    kind = "w8"
    wgroup = 0
    has_z = False
    try:
        from .quant_formats import (plan_engine_kind, read_quant_config,
                                    w4_native)
        plan = read_quant_config(model_dir)
        if plan["kind"] == "native":
            # our own formats: detect from tensor names (packed_weight -> W4)
            fs = [g for g in os.listdir(model_dir) if g.endswith(".safetensors")]
            kind = "w8"
            try:
                from safetensors import safe_open
                with safe_open(os.path.join(model_dir, fs[0]), framework="pt") as f:
                    kind = "w4" if any(k.endswith(".packed_weight")
                                       for k in f.keys()) else "w8"
            except Exception:
                pass
            wgroup = int(plan["raw"].get("group_size", 128)) if kind == "w4" else 0
        else:
            kind = plan_engine_kind(plan) or "w8"
            if kind == "w4":
                wgroup = plan["group"] if w4_native(plan) else 64
                if wgroup < 32:
                    wgroup = 64
                has_z = (plan["kind"] == "awq" and w4_native(plan)) or (
                    plan["kind"] == "gptq" and not plan["sym"]
                    and not plan["desc_act"])
    except Exception:
        pass

    def lin_bytes(n, nin, nout):
        if kind == "w4":
            scale = (nin // wgroup) * nout * 2 if wgroup else 0
            zeros = (nin // wgroup) * nout if (wgroup and has_z) else 0
            return n * (nin * nout // 2 + scale + zeros)
        return n * (nin * nout + nout * 4)
        return n * (nin * nout + nout * 4)

    # (count, in_features, out_features) per `weights.py` role table
    roles = [
        (n_lin, conv_dim, H), (n_lin, DV * NH, H), (n_lin, H, DV * NH),
        (n_full, NQ * HD * 2, H), (n_full, NKV * HD, H), (n_full, NKV * HD, H),
        (n_full, H, NQ * HD),
        (NL, I, H), (NL, I, H), (NL, H, I),
    ]
    per_layer_norm = H * 2
    weights = sum(lin_bytes(n, nin, nout) for n, nin, nout in roles)
    weights += n_full * NQ * HD * 2 * 2 + n_full * NKV * HD * 2 * 2   # attn_qn / attn_kn
    weights += n_lin * NH * H * 2 * 2                                # lin_b / lin_a
    weights += n_lin * conv_dim * conv_k * 2                         # lin_conv
    weights += n_lin * NH * 2 * 2                                    # lin_dt / lin_alog
    weights += n_lin * DV * 2                                        # lin_norm_w
    weights += 2 * NL * per_layer_norm                               # in/post norms
    weights += H * 2                                                 # final norm
    # embed int8 + per-row bf16 scale; head int8 [H, V] + per-row fp32 scale.
    # (A W4-packed head is implemented but shipped off by default: on the 0.8B
    # parity set it cost ~9 points of teacher-forced top-1 — 4-bit rows are too
    # coarse for logit reconstruction. See `head_w4=True` in Engine.)
    weights += V * H + V * 2
    weights += H * V + V * 4

    states = n_lin * NH * DK * DV * 4            # delta state fp32
    states += n_lin * 3 * conv_dim * 2           # conv state bf16
    kv = n_full * kv_seq * (2 * NKV * HD) * 2    # bf16 KV, full-attn layers only
    if os.environ.get("QWEN35_KV_INT8", "0") == "1":
        # int8 cache + per-(position, head, side) fp32 scales (see Engine)
        kv = n_full * kv_seq * ((2 * NKV * HD) + (2 * NKV) * 4)
    workspace = 64 * 1024 * 1024                 # prefill scratch, generous
    runtime_overhead = 256 * 1024 * 1024         # CUDA context, kernels, handles
    return weights + states + kv + workspace + runtime_overhead


def _request_vram(required, device):
    """Ask ComfyUI to evict only the shortfall, keeping resident models that fit."""
    free = mm.get_free_memory(device)
    shortfall = required + _VRAM_MARGIN - free
    if shortfall > 0:
        _log(f"[qwen35] need {required/1e9:.2f} GB, free {free/1e9:.2f} GB -> "
             f"freeing {shortfall/1e9:.2f} GB")
        mm.free_memory(shortfall, device)
        return True
    return False


def _get_engine(model_dir):
    """Load (or reuse) an engine, keeping at most one on the GPU at a time."""
    with _lock:
        eng = _cache.get(model_dir)
        if eng is not None:
            if not eng.on_gpu:
                eng.reload()
            return eng
        for key, other in list(_cache.items()):
            if key != model_dir:
                other.offload()
                del _cache[key]
        need = None
        if mm is not None:
            need = _estimate_engine_vram(model_dir)
            # Evict only what is missing so a diffusion model that still fits
            # next to the engine is never touched.
            _request_vram(need, torch.device("cuda"))
        try:
            eng = Engine(model_dir, device="cuda", log=_log)
        except torch.cuda.OutOfMemoryError:
            if mm is None:
                raise
            _log("[qwen35] out of VRAM, freeing everything and retrying")
            mm.free_memory(need or 4_000_000_000, torch.device("cuda"))
            torch.cuda.empty_cache()
            eng = Engine(model_dir, device="cuda", log=_log)
        _cache[model_dir] = eng
        return eng


def _get_tokenizer(model_dir):
    """Tokenizer is immutable and slow to build (20 MB vocab), so keep one per model."""
    with _lock:
        tok = _tok_cache.get(model_dir)
        if tok is None:
            tok = ChatTokenizer(model_dir)
            _tok_cache[model_dir] = tok
        return tok


def _log(msg):
    print(msg, flush=True)


def _stream_preview(tok, ids):
    """Push decoded partial text to the browser (streaming preview, 4.x).

    Uses ComfyUI's `PromptServer.send_sync` without a websocket sid, which
    broadcasts to connected clients; the web extension (web/) renders it on
    the node.  Silently no-ops when ComfyUI isn't running (e.g. in tests).
    """
    try:
        from comfy.cli_args import args as cli_args
        from server import PromptServer
    except Exception:
        try:
            from server import PromptServer  # type: ignore
        except Exception:
            return
    try:
        text = tok.decode(ids, skip_special_tokens=True)
        if text:
            PromptServer.instance.send_sync(
                "qwen35_preview", {"text": text, "stream_preview": True})
    except Exception:
        pass


class Qwen35PromptEnhancer:
    """Short text -> detailed prompt, via a local INT8 Qwen3.5 engine."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _model_dirs()
        if not models:
            models = ["(no models found in models/LLM)"]
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "a cat sitting on a windowsill",
                    "tooltip": "The short idea or keyword list to expand.",
                }),
                "model": (models, {
                    "tooltip": "Checkpoint folder under ComfyUI/models/LLM.",
                }),
                "memory_mode": (["offload_after_run", "keep_loaded"], {
                    "default": "offload_after_run",
                    "tooltip": "offload_after_run frees VRAM after every run; "
                               "keep_loaded keeps the engine resident for faster repeats.",
                }),
                "max_tokens": (MAX_TOKEN_CHOICES, {
                    "default": "1024",
                    "tooltip": "Upper bound on generated tokens.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": DEFAULT_SYSTEM,
                    "tooltip": "Instruction that defines the enhancer's style.",
                }),
                "target": (tuple(TARGETS.keys()), {
                    "default": "default",
                    "tooltip": "Downstream target: system/budget/headers "
                               "compile for what consumes the prompt (2.4).",
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "top_k": ("INT", {
                    "default": 20, "min": 0, "max": 200, "step": 1,
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Drop tokens below this fraction of the top "
                               "probability before top-p (0 disables).",
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Penalise tokens already generated (1.0 disables).",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Constant logit penalty for every token that "
                               "has appeared at least once (0 disables).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1,
                    "control_after_generate": True,
                }),
                "best_of": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Sample N candidates from one shared prefill and "
                               "keep the best-scoring one (2.2). N=1 is today's "
                               "single draw, and is much faster.",
                }),
                "stream_preview": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Push decoded partial text live to the browser "
                               "while generating (needs the web/ extension).",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "token_count", "report")
    FUNCTION = "enhance"
    CATEGORY = "text/Qwen3.5"
    OUTPUT_NODE = True

    def enhance(self, text, model, memory_mode, max_tokens, system_prompt,
                temperature=0.7, top_p=0.9, top_k=40, min_p=0.05,
                repetition_penalty=1.05, presence_penalty=0.0, seed=0,
                target="default", best_of=1, stream_preview=False):
        text = (text or "").strip()
        if not text:
            return {"ui": {"text": [""]}, "result": ("", 0, "")}

        tgt = get_target(target)
        model_dir = _resolve_model(model)
        limit = int(max_tokens)
        budget = tgt.get("budget")
        if budget:
            limit = min(limit, int(budget))
        eng = _get_engine(model_dir)
        tok = _get_tokenizer(model_dir)

        # System priority: explicit user text wins; else the target's own
        # system if one is defined and the field is untouched/empty; else the
        # node default.  (`tgt["system"]` is None for "default".)
        system = (system_prompt or "")
        if tgt.get("system") and system in ("", DEFAULT_SYSTEM):
            system = tgt["system"]
        if not system:
            system = DEFAULT_SYSTEM

        ids = tok.encode_chat(system, text)
        prefix_len = tok.encode_chat_prefix_len(system, ids)
        stop = tok.stop_ids
        mask = build_header_mask(tok, tgt.get("headers") or [], stop,
                                 eng.V, eng.device)
        newline_id = tok.token_to_id("\n")
        headers = tgt.get("headers") or []
        grammar = None
        if headers:
            grammar = SectionGrammar(headers, tok, eng.V, eng.device, stop,
                                     newline_id)
            if not grammar.uniform:
                # non-uniform tokenised lengths: keep the v1 header mask
                grammar = None
                _log("[qwen35] headers tokenise to non-uniform lengths; "
                     "using the v1 header mask for this target")

        pbar = comfy.utils.ProgressBar(limit) if comfy is not None else None
        stream = stream_preview and comfy is not None
        state = {"n": 0, "ids": []}

        def on_token(tok_id):
            state["n"] += 1
            if pbar is not None:
                pbar.update(1)
            if stream:
                state["ids"].append(tok_id)
                if state["n"] % 8 == 0:
                    _stream_preview(tok, state["ids"][-256:])

        gen_kw = dict(
            max_new_tokens=limit,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            seed=int(seed),
            stop_ids=stop,
            min_p=float(min_p),
            repetition_penalty=float(repetition_penalty),
            presence_penalty=float(presence_penalty),
            prefix_len=prefix_len,
            header_mask=mask,
            newline_id=newline_id,
            line_grammar=grammar,
        )
        try:
            with torch.inference_mode():
                if int(best_of) > 1:
                    rows = eng.generate_batch(ids, n=int(best_of), on_token=None,
                                              **gen_kw)
                    texts = [postprocess(tok.decode(r).strip(), tgt)
                             for r in rows]
                    winner, scores = score_candidates(
                        texts, keywords_of(text), budget=budget,
                        headers=tgt.get("headers") or None, tok=tok)
                    result = texts[winner]
                    token_count = len(tok.encode(result)) if result else 0
                    warnings = []
                    if budget and token_count > budget:
                        warnings.append(
                            f"best candidate is {token_count} tokens, over the "
                            f"{budget}-token target budget")
                    report = {"winner": winner, "n": len(texts),
                              "scores": scores, "warnings": warnings}
                    _log(f"[qwen35] winner={winner} "
                         + " ".join(f"{s['index']}:{s['score']}" for s in scores))
                else:
                    out_ids, warnings = eng.generate(
                        ids, on_token=on_token, **gen_kw)
                    result = postprocess(tok.decode(out_ids).strip(), tgt)
                    token_count = len(tok.encode(result)) if result else 0
                    if budget and token_count > budget:
                        warnings.append(
                            f"generated prompt is {token_count} tokens, over the "
                            f"{budget}-token target budget")
                    report = {"winner": 0, "n": 1, "scores": [],
                              "warnings": warnings}
            for w in warnings:
                _log(f"[qwen35] warning: {w}")
        finally:
            if memory_mode == "offload_after_run":
                with _lock:
                    eng.offload()

        import json
        warn_txt = json.dumps(report, ensure_ascii=False, sort_keys=True)
        return {"ui": {"text": [result, *warnings]},
                "result": (result, token_count, warn_txt)}


NODE_CLASS_MAPPINGS = {
    "Qwen35PromptEnhancer": Qwen35PromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen35PromptEnhancer": "Qwen3.5 Prompt Enhancer (INT8)",
}
