"""Checkpoint loading -> stacked tensors ready for the Triton kernels.

Three groups of checkpoint formats are supported for the stacked linear layers:

* **W8A16** (our own format) — `<proj>.qweight` int8 `[in, out]` + `<proj>.scale`
  fp32 `[out]`.  Weights stay int8 and are used directly by the GEMM.
* **W4A16 groupwise** (our own format) — `<proj>.packed_weight` uint8
  `[out, in//2]` (even-K low nibble / odd-K high nibble) + `<proj>.scale` bf16
  `[out, in//group]`.  Weights stay packed at 4 bits and are expanded to int8
  inside the dot loop, so a 2B model keeps 685 MB of weights instead of 1.37 GB.
* **AWQ / GPTQ / compressed-tensors** (`quantization_config` with a
  `quant_method` other than ours) — converted at load time into the layouts
  above; see `quant_formats.py`.  Checkpoints whose stored grid maps directly
  onto the kernels are repacked bit-exactly (symmetric CT int4 at a supported
  group size, symmetric GPTQ int4, AutoAWQ asym int4 via the per-group-zero
  kernel path); anything else is dequantized and requantized (group 16/act-
  order/asymmetric CT/int8-with-group-scales), which adds one quantisation
  round on top of the checkpoint's own.

The embedding table is always quantised at load time (int8, per-vocab-row
scale; W4 head optional via `head_w4=True`, default off) regardless of format.
"""
import glob
import hashlib
import json
import os

import torch

from .quant_formats import (CheckpointReader, convert_proj, plan_engine_kind,
                            read_quant_config, requant_w4_sym, requant_w8_row,
                            w4_native)


# --------------------------------------------------------------------------
# pack cache: the AWQ/GPTQ/CT decode + requantize at load time is
# deterministic for a given checkpoint and converter version, so the stacked
# tensors are dumped once and memory-mapped afterwards.  The key covers the
# checkpoint files, the converter code and the embedding options, so edits to
# the converters or swapping checkpoints force a reconvert.
# --------------------------------------------------------------------------

_PACK_FORMAT = 1


def _pack_root():
    override = os.environ.get("QWEN35_WEIGHTS_CACHE")
    root = override or (
        os.path.join(os.environ.get("LOCALAPPDATA",
                                    os.path.join(os.path.expanduser("~"),
                                                 "AppData", "Local")),
                     "qwen35_prompt_enhancer", "weights_cache")
        if os.name == "nt" else
        os.path.join(os.path.expanduser("~"), ".cache",
                     "qwen35_prompt_enhancer", "weights_cache"))
    try:
        os.makedirs(root, exist_ok=True)
        probe = os.path.join(root, ".probe")
        with open(probe, "wb") as f:
            f.write(b"ok")
        os.remove(probe)
    except OSError:
        return None
    return root


def _pack_key(model_dir, embed_bf16, head_w4):
    sig = []
    for pattern in ("*.safetensors", "*.index.json"):
        for f in sorted(glob.glob(os.path.join(model_dir, pattern))):
            try:
                st = os.stat(f)
                sig.append((os.path.relpath(f, model_dir), st.st_size,
                            st.st_mtime_ns))
            except OSError:
                return None
    try:
        st = os.stat(os.path.join(model_dir, "config.json"))
        sig.append(("config.json", st.st_size, st.st_mtime_ns))
    except OSError:
        return None
    # converter version: touching these files forces a reconvert
    for extra in ("weights.py", "quant_formats.py"):
        try:
            st = os.stat(os.path.join(os.path.dirname(__file__), extra))
            sig.append((extra, st.st_size, st.st_mtime_ns))
        except OSError:
            return None
    sig.append(("options", bool(embed_bf16), bool(head_w4), _PACK_FORMAT))
    blob = json.dumps(sig, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()


def _save_pack(w, pack):
    tmp = f"{pack}.tmp-{os.getpid()}"
    state = {
        "format": _PACK_FORMAT,
        "quant": w.quant,
        "has_z": w.has_z,
        "cfg": w.cfg,
        "tensors": {k: v.detach().cpu().contiguous() for k, v in w.t.items()},
    }
    torch.save(state, tmp)
    os.replace(tmp, pack)


def _load_pack(pack, log):
    state = torch.load(pack, map_location="cpu", weights_only=False)
    if state.get("format") != _PACK_FORMAT:
        log("[qwen35] pack cache: format mismatch; reconverting")
        return None
    w = QwenWeights(state["cfg"], quant=state.get("quant", "w8"))
    w.has_z = bool(state.get("has_z", False))
    w.t = {k: v.contiguous() for k, v in state["tensors"].items()}
    w.device = torch.device("cpu")
    return w


class QwenWeights:
    """All quantised weights stacked per role (layer index on dim 0)."""

    def __init__(self, cfg, quant="w8"):
        self.cfg = cfg
        self.quant = quant
        self.has_z = False        # per-group asymmetric zeros present (W4)
        self.t = {}               # name -> tensor
        self.device = torch.device("cpu")
        self._cpu_backup = None

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, k):
        return self.t[k]

    def to(self, device):
        if self.device == torch.device(device):
            return self
        for k, v in self.t.items():
            self.t[k] = v.to(device, non_blocking=False)
        # reload path should leave storage on plain CUDA or plain CPU tensors
        # again — dropping pinning keeps the allocator from tracking it as
        # "still pinned" when the same object is re-used for the next switch
        self.device = torch.device(device)
        return self

    def offload(self):
        """Move every weight tensor to pinned CPU memory, freeing VRAM."""
        if self.device.type == "cpu" and all(v.device.type == "cpu"
                                             for v in self.t.values()):
            return
        for k, v in self.t.items():
            try:
                self.t[k] = v.to("cpu", non_blocking=False).pin_memory()
            except Exception:
                self.t[k] = v.to("cpu")
        self.device = torch.device("cpu")
        torch.cuda.empty_cache()

    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self.t.values())


def _stacked(shape, dtype, device):
    return torch.empty(shape, dtype=dtype, device=device)


def _is_qwen35_dir(d):
    cfgp = os.path.join(d, "config.json")
    try:
        cfg = json.load(open(cfgp, encoding="utf-8"))
    except Exception:
        return False
    sub = cfg.get("text_config", cfg)
    mt = str(sub.get("model_type", ""))
    if "qwen3_5" not in mt and "qwen3.5" not in mt:
        return False
    return "hidden_size" in sub and "layer_types" in sub


def discover_models(roots):
    """Quantised Qwen3.5 checkpoints under `roots` -> [(label, path), ...].

    Accepts our own W8A16/W4A16 folders as well as AWQ / GPTQ /
    compressed-tensors folders, sharded or not.
    """
    if isinstance(roots, str):
        roots = [roots]
    out, seen = [], {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name)
            if not os.path.isdir(d) or name.startswith("."):
                continue
            if not _is_qwen35_dir(d):
                continue
            if not glob.glob(os.path.join(d, "*.safetensors")):
                continue
            label = name
            if label in seen:
                label = f"{os.path.basename(os.path.normpath(root))}/{name}"
            seen[label] = d
            out.append((label, d))
    return out


def load_weights(model_dir, device="cuda", log=print, embed_bf16=False,
                 head_w4=False):
    target = torch.device(device)
    use_pack = os.environ.get("QWEN35_DISABLE_PACK") != "1"
    root = key = pack = None
    if use_pack:
        root = _pack_root()
        key = _pack_key(model_dir, embed_bf16, head_w4)
        pack = os.path.join(root, f"{key}.pt") if (root and key) else None
        if pack and os.path.exists(pack):
            try:
                w = _load_pack(pack, log)
                if w is not None:
                    w.to(target)
                    log("[qwen35] engine weights loaded from pack cache")
                    return w
            except Exception as e:
                log(f"[qwen35] pack cache unreadable ({e}); reconverting")
    # build on the target device; the pack dump keeps CPU copies, so cached
    # and fresh loads hold the same bits
    build_dev = target
    cfg = json.load(open(os.path.join(model_dir, "config.json"),
                         encoding="utf-8"))
    if "text_config" in cfg:
        cfg = cfg["text_config"]
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
    KD = cfg["linear_key_dim"] if "linear_key_dim" in cfg else NH * DK
    VD = DV * NH
    NQ = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    conv_dim = 2 * KD + VD

    plan = read_quant_config(model_dir)
    reader = CheckpointReader(model_dir)
    keys = reader.keys()
    # checkpoints from full-model repos nest the text weights under a prefix
    prefix = "model.language_model." if any(
        k.startswith("model.language_model.") for k in keys) else "model."

    if plan["kind"] == "native":
        packed = any(k.endswith(".packed_weight") for k in keys)
        kind = "w4" if packed else "w8"
        wgroup = int(plan["raw"].get("group_size", 128)) if packed else 0
        src_set = {kind + "-custom"}
    else:
        kind = plan_engine_kind(plan)
        if kind == "w4":
            if w4_native(plan):
                wgroup = plan["group"]
            else:
                wgroup = 32      # requant target: finest native group
        else:
            wgroup = 0
        src_set = set()
    # asymmetric per-group zeros: only when the checkpoint's native asym grid
    # is used directly (the requant path quantises symmetrically)
    has_z = (kind == "w4"
             and ((plan["kind"] == "awq" and w4_native(plan))
                  or (plan["kind"] == "gptq" and not plan["sym"]
                      and not plan["desc_act"] and w4_native(plan)))
             and wgroup >= 32)

    dev = build_dev
    w = QwenWeights(cfg, quant=kind)
    w.has_z = has_z
    t = w.t

    # ---- quantised linear layers: (role, out_dim, in_dim, n_layers) ---------
    roles = [
        ("lin_qkv", conv_dim, H, n_lin),
        ("lin_z", VD, H, n_lin),
        ("lin_out", H, VD, n_lin),
        ("attn_q", NQ * HD * 2, H, n_full),
        ("attn_k", NKV * HD, H, n_full),
        ("attn_v", NKV * HD, H, n_full),
        ("attn_o", H, NQ * HD, n_full),
        ("mlp_gate", I, H, NL),
        ("mlp_up", I, H, NL),
        ("mlp_down", H, I, NL),
    ]
    if kind == "w4":
        for role, nout, nin, n in roles:
            assert nin % wgroup == 0 and nin % 2 == 0, (role, nin, wgroup)
            # K-major: [K//2, N] nibbles and [K//wg, N] scales, so the GEMM
            # reads each tile as one coalesced run along N.
            t[role] = _stacked((n, nin // 2, nout), torch.uint8, dev)
            t[role + "_s"] = _stacked((n, nin // wgroup, nout), torch.bfloat16, dev)
            if has_z:
                t[role + "_z"] = _stacked((n, nin // wgroup, nout), torch.uint8, dev)
    else:
        for role, nout, nin, n in roles:
            t[role] = _stacked((n, nin, nout), torch.int8, dev)
            t[role + "_s"] = _stacked((n, nout), torch.float32, dev)

    t["final_norm"] = _stacked((H,), torch.bfloat16, dev)
    t["in_norm"] = _stacked((NL, H), torch.bfloat16, dev)
    t["post_norm"] = _stacked((NL, H), torch.bfloat16, dev)
    t["lin_b"] = _stacked((n_lin, NH, H), torch.bfloat16, dev)
    t["lin_a"] = _stacked((n_lin, NH, H), torch.bfloat16, dev)
    t["lin_conv"] = _stacked((n_lin, conv_dim, cfg["linear_conv_kernel_dim"]),
                             torch.bfloat16, dev)
    t["lin_dt"] = _stacked((n_lin, NH), torch.bfloat16, dev)
    t["lin_alog"] = _stacked((n_lin, NH), torch.bfloat16, dev)
    t["lin_norm_w"] = _stacked((n_lin, DV), torch.bfloat16, dev)
    t["attn_qn"] = _stacked((n_full, HD), torch.bfloat16, dev)
    t["attn_kn"] = _stacked((n_full, HD), torch.bfloat16, dev)

    # ---- per-projection emission ------------------------------------------
    ext = plan["kind"] != "native"

    def emit_w4(role, slot, conv):
        """Engine W4 pieces -> stacked tensors for one projection."""
        if conv["fmt"] == "nib":
            t[role][slot].copy_(conv["nib_T"].to(dev, non_blocking=True))
            t[role + "_s"][slot].copy_(
                conv["scale_T"].to(torch.bfloat16).to(dev, non_blocking=True))
        elif conv["fmt"] == "nibz":
            t[role][slot].copy_(conv["nib_T"].to(dev, non_blocking=True))
            t[role + "_s"][slot].copy_(
                conv["scale_T"].to(torch.bfloat16).to(dev, non_blocking=True))
            t[role + "_z"][slot].copy_(conv["zero_T"].to(dev, non_blocking=True))
        elif conv["fmt"] == "w":
            packed, sc = requant_w4_sym(conv["w"].float())
            t[role][slot].copy_(packed.to(dev, non_blocking=True))
            t[role + "_s"][slot].copy_(sc.to(dev, non_blocking=True))
        else:  # dense bf16 [out, in]
            packed, sc = requant_w4_sym(
                conv["weight"].t().float().contiguous())
            t[role][slot].copy_(packed.to(dev, non_blocking=True))
            t[role + "_s"][slot].copy_(sc.to(dev, non_blocking=True))
        src_set.add(conv.get("src", "?"))

    def emit_w8(role, slot, conv):
        if conv["fmt"] == "dense":
            q, sc = requant_w8_row(conv["weight"].t().float().contiguous())
        else:
            q, sc = requant_w8_row(conv["w"].float())
        t[role][slot].copy_(q.to(dev, non_blocking=True))
        t[role + "_s"][slot].copy_(sc.to(dev, non_blocking=True))
        src_set.add(conv.get("src", conv["fmt"]))

    def put(role, slot, fq):
        conv = convert_proj(reader, fq, plan) if ext else None
        if conv is None:  # our own formats, stored natively
            if kind == "w4":
                t[role][slot].copy_(
                    reader.get(fq + ".packed_weight").t().contiguous()
                    .to(dev, non_blocking=True))
                t[role + "_s"][slot].copy_(
                    reader.get(fq + ".scale").t().contiguous()
                    .to(dev, non_blocking=True))
            else:
                t[role][slot].copy_(reader.get(fq + ".qweight")
                                    .to(dev, non_blocking=True))
                t[role + "_s"][slot].copy_(reader.get(fq + ".scale").float()
                                           .to(dev, non_blocking=True))
        elif kind == "w4":
            emit_w4(role, slot, conv)
        else:
            emit_w8(role, slot, conv)

    lin_i = full_i = 0
    for i in range(NL):
        if layer_types[i] == "linear_attention":
            p = f"{prefix}layers.{i}.linear_attn."
            put("lin_qkv", lin_i, p + "in_proj_qkv")
            put("lin_z", lin_i, p + "in_proj_z")
            put("lin_out", lin_i, p + "out_proj")
            t["lin_b"][lin_i].copy_(reader.get(p + "in_proj_b.weight").to(dev))
            t["lin_a"][lin_i].copy_(reader.get(p + "in_proj_a.weight").to(dev))
            t["lin_conv"][lin_i].copy_(
                reader.get(p + "conv1d.weight").squeeze(1).to(dev))
            t["lin_dt"][lin_i].copy_(reader.get(p + "dt_bias").to(dev))
            t["lin_alog"][lin_i].copy_(reader.get(p + "A_log").to(dev))
            t["lin_norm_w"][lin_i].copy_(reader.get(p + "norm.weight").to(dev))
            lin_i += 1
        else:
            p = f"{prefix}layers.{i}.self_attn."
            put("attn_q", full_i, p + "q_proj")
            put("attn_k", full_i, p + "k_proj")
            put("attn_v", full_i, p + "v_proj")
            put("attn_o", full_i, p + "o_proj")
            t["attn_qn"][full_i].copy_(reader.get(p + "q_norm.weight").to(dev))
            t["attn_kn"][full_i].copy_(reader.get(p + "k_norm.weight").to(dev))
            full_i += 1

        p = f"{prefix}layers.{i}."
        put("mlp_gate", i, p + "mlp.gate_proj")
        put("mlp_up", i, p + "mlp.up_proj")
        put("mlp_down", i, p + "mlp.down_proj")
        t["in_norm"][i].copy_(reader.get(p + "input_layernorm.weight").to(dev))
        t["post_norm"][i].copy_(reader.get(
            p + "post_attention_layernorm.weight").to(dev))

    # ---- embedding table + lm_head -----------------------------------------
    # Both formats tie the head to `embed_tokens.weight`, but the two consumers
    # need different layouts: the gather wants [V, H] and the logits GEMM wants
    # [H, V]. Two tensors: embed int8 with one scale per vocabulary row, head
    # int8 (or optional W4) with per-row scales.
    src = reader.get(prefix + "embed_tokens.weight")
    if embed_bf16:
        t["embed"] = _stacked((V, H), torch.bfloat16, dev)
        t["embed"].copy_(src.to(dev, non_blocking=False))
    else:
        t["embed"] = _stacked((V, H), torch.int8, dev)
        t["embed_s"] = _stacked((V,), torch.bfloat16, dev)
        step = 8192
        for s0 in range(0, V, step):
            s1 = min(V, s0 + step)
            blk = src[s0:s1].to(dev, non_blocking=False).float()
            amax = blk.abs().amax(dim=1).clamp_min(1e-8)
            sc = amax / 127.0
            t["embed"][s0:s1].copy_(
                torch.clamp(torch.round(blk / sc[:, None]), -127, 127)
                .to(torch.int8))
            t["embed_s"][s0:s1].copy_(sc.to(torch.bfloat16))
            del blk

    head_name = "lm_head.weight" if reader.has("lm_head.weight") else None
    head_src = reader.get(head_name) if head_name else src
    if head_w4:
        # W4, K-major nibbles so `gemm_i8` dispatches to `gemm_w4`.
        # NOT the default: the 4-bit grid measurably regresses teacher-forced
        # top-1 (see README).
        head_q = _stacked((H // 2, V), torch.uint8, dev)
        head_s = _stacked((H // 64, V), torch.bfloat16, dev)  # per-64 group
        step = 8192
        for s0 in range(0, V, step):
            s1 = min(V, s0 + step)
            blk = head_src[s0:s1].to(dev, non_blocking=False).float()
            n, _ = blk.shape
            b3 = blk.view(n, H // 64, 64)
            amax = b3.abs().amax(dim=2).clamp_min(1e-8)
            sc = amax / 7.0                            # symmetric [-8, 7] int4
            q = torch.clamp(torch.round(b3 / sc[:, :, None]), -8, 7) + 8
            lo = q[:, :, 0::2].to(torch.uint8)         # even K in the low nibble
            hi = q[:, :, 1::2].to(torch.uint8)         # odd K in the high nibble
            head_q[:, s0:s1].copy_(
                (lo.to(torch.int32) | (hi.to(torch.int32) << 4))
                .to(torch.uint8).reshape(n, H // 2).t().contiguous())
            head_s[:, s0:s1].copy_(sc.t().contiguous())
            del blk, q, lo, hi
        t["lm_head"] = head_q
        t["lm_head_s"] = head_s
    else:
        # int8 [H, V] with one scale per vocab row
        head_q = _stacked((H, V), torch.int8, dev)
        head_s = _stacked((V,), torch.float32, dev)
        step = 8192
        for s0 in range(0, V, step):
            s1 = min(V, s0 + step)
            blk = head_src[s0:s1].to(dev, non_blocking=False).float()
            amax = blk.abs().amax(dim=1).clamp_min(1e-8)
            sc = amax / 127.0
            head_q[:, s0:s1].copy_(
                torch.clamp(torch.round(blk / sc[:, None]), -127, 127)
                .to(torch.int8).t().contiguous())
            head_s[s0:s1].copy_(sc)
            del blk
        t["lm_head"] = head_q
        t["lm_head_s"] = head_s

    t["final_norm"].copy_(reader.get(prefix + "norm.weight").to(dev))

    dense_n = sum(1 for s in src_set if s.startswith("dense"))
    req = sorted(s for s in src_set if "requant" in s)
    log(f"[qwen35] weights loaded: {w.nbytes()/1e9:.2f} GB "
        f"({','.join(sorted(x for x in src_set
                           if not x.startswith('dense')))}"
        f", embed {'bf16' if embed_bf16 else 'int8'}, "
        f"head {'W4' if head_w4 else 'int8'}, "
        f"zeros {'on' if has_z else 'off'}) on {device}")
    if pack is not None:
        try:
            _save_pack(w, pack)
            log(f"[qwen35] cached packed weights to {pack}")
        except Exception as e:
            log(f"[qwen35] pack cache write failed (continuing): {e}")
    if w.device != target:
        w.to(target)
    w.device = target
    if req:
        log(f"[qwen35] note: requantised from native grid: {req[0]}"
            + (f" (+{len(req)-1} projections)" if len(req) > 1 else ""))
    if dense_n:
        log(f"[qwen35] note: {dense_n} non-quantised projection"
            f"{'s' if dense_n != 1 else ''} from the checkpoint (ignore list), "
            f"quantised to int8/int4 at load")
    return w
