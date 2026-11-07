"""Third-party quant formats -> engine layouts.

Converts AWQ / GPTQ / compressed-tensors ("pack-quantized") checkpoints into the
two stacked encodings our Triton kernels consume:

* engine W4: K-major packed nibbles `uint8 [K//2, N]` (even K low nibble, odd K
  high nibble, value = nibble - 8) + bf16 scales `[K//wg, N]` (plus an optional
  per-group zero `uint8 [K//wg, N]`, decode value = nibble - z);
* engine W8: int8 `[K, N]` + per-output-column fp32 scales `[N]`.

Decode formulas come from the format owners:

* compressed-tensors `pack-quantized` (`neuralmagic/compressed-tensors`,
  `compressors/pack_quantized/base.py` + `helpers.py`): weight is stored `[N,
  ceil(K*bits/32)]` int32, packed densely along K; the stored nibble is
  `signed + 1<<(bits-1)` (offset applied for every int width, wrap-around for
  8 bits), so `q = nibble - offset`.  Symmetric schemes drop the zero point, so
  a symmetric CT int4 checkpoint repacks into the engine layout bit-exactly —
  only byte order and scale transposition change.
* GPTQ / AutoGPTQ (`AutoGPTQ/nn_modules/triton_utils/dequant.py`): qweight
  int32 `[K//EPF, N]`, straight nibble order, `EPF = 32//bits`; qzeros int32
  `[G, N//EPF]` decoded the same way; effective zero = stored + 1 and
  `w = (q - z_eff) * scale`.  A symmetric GPTQ checkpoint stores `z = 7` in
  every nibble, reducing the decode to the same `nibble - 8` grid.
* AutoAWQ (`casper-hansen/AutoAWQ`, `awq/utils/packing_utils.py`): qweight
  int32 `[K, N//EPF]`, nibble order within each 8-nibble group fixed by
  `AWQ_REVERSE_ORDER`; zeros are used *directly* (no `+1`);
  `w = (q - z) * scale`; scales repeat_interleaved along K.

Where the stored grid cannot map onto the engine kernels (unsupported group
size, act-order / desc_act, asymmetric CT, int8 with group scales), the loader
falls back to dequantize -> requantize; `convert_proj` reports what it did so
the caller can surface it.
"""
import glob
import json
import os

import torch
from safetensors import safe_open

# Weight-scale group sizes the engine W4 kernels represent natively:
# 32 (two SUB=32 dots per block) or 64 (and any multiple of 64).  Group 16
# cannot drive tl.dot (K >= 32 required) -> requant.
_NAT_W4 = {32, 64} | {m for m in range(64, 4097, 64)}


def native_w4_group(wg):
    return wg in _NAT_W4


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def read_quant_config(model_dir):
    """Parse quantization_config from config.json (+ optional standalone file)."""
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    qcfg = dict(cfg.get("quantization_config") or {})
    p = os.path.join(model_dir, "quantization_config.json")
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                qcfg = {**qcfg, **json.load(f)}
        except Exception:
            pass
    method = str(qcfg.get("quant_method", ""))
    plan = {"kind": "native", "bits": None, "group": None, "sym": True,
            "desc_act": False, "ignore": [], "raw": qcfg}

    if method == "gptq" or method.startswith("auto_round") or method in ("autoround", "auto-round"):
        # GPTQ and AutoRound (HF AutoRoundGPTQ packing: `auto_round:auto_gptq`)
        plan["kind"] = "gptq"
        plan["bits"] = int(qcfg.get("bits", 4))
        plan["group"] = int(qcfg.get("group_size", -1))
        plan["sym"] = bool(qcfg.get("sym", True)) if "sym" in qcfg else \
            bool(qcfg.get("symmetric", True))
        plan["desc_act"] = bool(qcfg.get("desc_act", False))
        plan["ignore"] = list(qcfg.get("excluded_modules")
                              or qcfg.get("modules_to_not_convert")
                              or qcfg.get("ignore") or [])
    elif method == "awq" or method == "awq_gemm":
        plan["kind"] = "awq"
        plan["bits"] = int(qcfg.get("bits", 4))
        plan["group"] = int(qcfg.get("group_size", -1))     # -1 = per-channel
        # legacy AutoAWQ is asymmetric by construction (it always stores zeros)
        plan["sym"] = not bool(qcfg.get("zero_point", True))
        plan["ignore"] = list(qcfg.get("modules_to_not_convert")
                              or qcfg.get("ignore")
                              or qcfg.get("excluded_modules") or [])
    elif (method == "compressed-tensors"
          or str(qcfg.get("format", "")) == "pack-quantized"):
        plan["kind"] = "compressed-tensors"
        spec = None
        for g in (qcfg.get("config_groups") or {}).values():
            if isinstance(g, dict) and g.get("weights"):
                spec = g["weights"]
                break
        if spec is None:
            spec = qcfg.get("weights") or {}
        plan["bits"] = int(spec.get("num_bits", 4))
        plan["strategy"] = str(spec.get("strategy", "group"))
        gs = spec.get("group_size")
        plan["group"] = int(gs) if gs is not None else None
        plan["sym"] = bool(spec.get("symmetric", True))
        plan["ignore"] = list(qcfg.get("ignore") or [])
    return plan


def plan_engine_kind(plan):
    if plan["kind"] == "native":
        return None                     # decided from tensor dtypes
    return "w4" if plan["bits"] == 4 else "w8"


def _is_ignored(plan, name):
    return any(name.startswith(pat) or name == pat
               for pat in (plan.get("ignore") or []))


def w4_native(plan):
    """True if the checkpoint's W4 grid maps directly onto engine kernels."""
    if plan["kind"] == "native":
        return True
    wg = plan["group"]
    return (not plan["desc_act"]
            and wg is not None and wg > 0 and native_w4_group(wg))


# --------------------------------------------------------------------------
# checkpoint reader (single or sharded safetensors)
# --------------------------------------------------------------------------

class CheckpointReader:
    """Reads tensors from one or several safetensors shards of a model dir."""

    def __init__(self, model_dir):
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors in {model_dir}")
        self.dir = model_dir
        if len(files) == 1:
            self.solo = files[0]
            self.map = None
        else:
            idx_files = [f for f in os.listdir(model_dir)
                         if f.endswith(".index.json")]
            if idx_files:
                with open(os.path.join(model_dir, idx_files[0]),
                          encoding="utf-8") as f:
                    self.map = dict(json.load(f).get("weight_map", {}))
            else:
                # rare: several shards without an index; scan keys per file
                self.map = {}
                for fpath in files:
                    with safe_open(fpath, framework="pt") as f:
                        for k in f.keys():
                            self.map[k] = fpath
            self.solo = None
        self.handles = {}

    def _handle(self, path):
        h = self.handles.get(path)
        if h is None:
            h = safe_open(path, framework="pt")
            self.handles[path] = h
        return h

    @property
    def solo_keys(self):
        if not hasattr(self, "_solo_keys"):
            self._solo_keys = set(self._handle(self.solo).keys())
        return self._solo_keys

    def has(self, name):
        if self.solo is not None:
            return name in self.solo_keys
        return name in self.map

    def get(self, name):
        if self.solo is not None:
            return self._handle(self.solo).get_tensor(name)
        path = self.map[name]
        if not os.path.isabs(path):
            path = os.path.join(self.dir, path)
        return self._handle(path).get_tensor(name)

    def keys(self):
        if self.solo is not None:
            return set(self.solo_keys)
        return set(self.map)


# --------------------------------------------------------------------------
# low-level decoders
# --------------------------------------------------------------------------

def _nibbles(qw, bits, device=None):
    """int32 packed [R, W] -> uint8 nibbles [R, W*EPF], straight bit order.

    Eq. AutoGPTQ dequant: q(k, n) = (qw[k // EPF, n] >> ((k % EPF) * bits)) & mask.
    """
    epf = 32 // bits
    q = qw.to(torch.int64) if qw.numel() * epf < 2 ** 31 else qw
    shifts = torch.arange(epf, dtype=q.dtype, device=q.device) * bits
    n = ((q.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).to(torch.uint8)
    return n.reshape(n.shape[0], -1)        # lex (w, j): idx = w*epf + j


def _deinterleave(nib, bits):
    """Restore AutoAWQ's permuted nibble order along the last axis.

    `reverse_awq_order` rewrites out[:, c] = in[:, REV[c]] with
    REV = [0, 4, 1, 5, 2, 6, 3, 7] per 8-nibble group, so the same permutation
    applied again restores the natural sequence.
    """
    rev = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7][:32 // bits],
                       device=nib.device)
    return nib.index_select(-1, rev)


def _pack_pairs(nib):
    """uint8 [K, N] nibbles -> [K//2, N] packed, even K low / odd K high."""
    K = nib.shape[0] - (nib.shape[0] % 2)
    nib = nib[:K]
    lo = nib[0::2].to(torch.int32)
    hi = nib[1::2].to(torch.int32)
    return (lo | (hi << 4)).to(torch.uint8)


def _sanity(w, name):
    """Guards against a wrong decode convention (e.g. wrong zero offset)."""
    m = w.mean().item()
    s = w.std().item() + 1e-9
    if abs(m) > 0.25 * s:
        raise RuntimeError(
            f"{name}: decoded weights look wrong (|mean| {abs(m):.4g} vs "
            f"std {s:.4g}) - decode formula/offset likely mismatched")


# --------------------------------------------------------------------------
# engine-side requantization (also used for dense bf16 projections)
# --------------------------------------------------------------------------

def _pairs_from(n3):
    """nibbles [K, N] uint8/int32 -> packed [K//2, N] (even low, odd high)."""
    lo = n3[0::2].to(torch.int32)
    hi = n3[1::2].to(torch.int32)
    return (lo | (hi << 4)).to(torch.uint8)


def requant_w4_sym(w, group=32):
    """float [K, N] -> (packed [K//2, N] uint8, scales [K//group, N] bf16).

    Symmetric int4, per-(group, n) scale, engine grid [-8, 7] + 8.  The target
    group is 32 when possible (the finest grid the kernels represent natively;
    ideal for requantizing fine-grained GPTQ g16 checkpoints) or 64.
    """
    K, N = w.shape
    assert group in (32, 64) and K % group == 0, (K, group)
    q4 = w.view(K // group, group, N)
    sc = q4.abs().amax(dim=1).clamp_min(1e-8) / 7.0
    q = torch.clamp(torch.round(q4 / sc[:, None, :]), -8, 7) + 8
    sc = sc.to(torch.bfloat16)
    packed = _pairs_from(q.reshape(K, N))
    return packed, sc


def requant_w8_row(w):
    """float [K, N] -> (int8 [K, N], fp32 scale [N]) per-column symmetric."""
    amax = w.abs().amax(dim=0).clamp_min(1e-8)          # [N]
    sc = (amax / 127.0).float()
    q = torch.clamp(torch.round(w / sc[None, :]), -127, 127).to(torch.int8)
    return q, sc


# --------------------------------------------------------------------------
# per-format decoders -> engine orientation ([K, N]; N = out features)
# --------------------------------------------------------------------------

def decode_ct(raw, plan, name):
    """compressed-tensors pack-quantized -> engine pieces.

    weight [out, in] = [N, K]; weight_packed int32 [N, ceil(K*bits/32)];
    scales [N, G] (group strategy).  Returns:
      sym int4 native groups: {"fmt": "nib", "nib_T": [K//2, N] uint8,
          "scale_T": [K//wg, N] float, "group": wg}
      otherwise: {"fmt": "w", "w": [K, N] float32} for requantization.
    """
    bits = plan["bits"]
    sym = plan["sym"]
    wg = plan["group"]
    strategy = plan.get("strategy", "group")
    qw = raw["weight_packed"]                             # int32 [N, W]
    qs = raw["weight_scale"]
    shape = raw.get("weight_shape")
    N = qw.shape[0]
    W = qw.shape[1]
    offset = 1 << (bits - 1)

    if shape is not None:
        Nc, K = [int(x) for x in shape.tolist()]
    else:
        K = W * (32 // bits)

    nib = _nibbles(qw, bits) if bits != 8 else None   # uint8 [N, K] for <8-bit

    if strategy == "group" and wg:
        if qs.shape != (N, K // wg) and qs.shape == (K // wg, N):
            qs = qs.t()
        assert qs.shape == (N, K // wg), (name, tuple(qs.shape), (N, K // wg))
        if bits == 8:
            # one byte per element; symmetric CT stores signed+128 (uint8)
            q = qw.view(torch.uint8).reshape(N, W * 4)[:, :K].to(torch.float32)
            q -= offset                                   # -128 for 8-bit
        else:
            nib = nib[:, :K]
            q = nib.to(torch.float32) - offset
        s = qs.to(torch.float32)
        sc = s.repeat_interleave(wg, dim=1)[:, :K]        # [N, K]
        w = q * sc
        if not sym:
            zp = raw.get("weight_zero_point")
            if zp is None:
                raise RuntimeError(f"{name}: asymmetric CT but no zero_point")
            # zero_point is stored with the same dense packing as the weights
            # (offset-encoded integers, packed along K with packed_dim=0), so
            # it decodes to one integer per (group, N) -> shape [N, K//wg].
            zb = zp.view(torch.uint8) if zp.dtype == torch.int32 else zp
            zq = (zb.reshape(zb.shape[0], -1)[:, :K // wg].to(torch.float32)
                  - offset)
            w = (q - zq.repeat_interleave(wg, dim=1)[:, :K]) * sc
    else:
        if bits != 8:
            nib = nib[:, :K]
        q = (nib.to(torch.float32) - offset) if bits != 8 else (
            qw.view(torch.uint8).reshape(N, -1)[:, :K].to(torch.float32) - offset)
        sc = qs.to(torch.float32)
        if sc.shape == (N,):
            sc = sc[:, None]
        sc = sc.repeat_interleave(K, dim=1)[:, :K] if sc.shape[1] == 1 else sc
        w = q * sc

    _sanity(w, name)
    if sym and bits == 4 and w4_native(plan):
        # Native repack: nibbles are already on the engine grid (value = nib-8)
        packed = _pairs_from(nib.t().contiguous())        # [K//2, N]
        return {"fmt": "nib", "nib_T": packed, "scale_T": s.t().contiguous(),
                "group": wg, "src": f"ct-sym-g{wg}"}
    return {"fmt": "w", "w": w.t().contiguous(), "requant": "ct-grid",
            "src": f"ct-{strategy}-g{wg}" + ("" if sym else "-asym")}


def decode_gptq(raw, plan, name):
    """AutoGPTQ decode (also matches AutoRound-quantized GPTQ checkpoints)."""
    bits = plan["bits"]
    qw = raw["qweight"]                                   # int32 [K//EPF, N]
    qs = raw["scale"].to(torch.float32)
    kz = raw.get("qzeros")
    g_idx = raw.get("g_idx")
    sym = plan["sym"]
    wg = plan["group"]
    EPF = 32 // bits
    K = qw.shape[0] * EPF
    N = qw.shape[1]
    assert K % wg == 0, (name, K, wg)
    G = K // wg
    # each int32 word packs EPF K-nibbles for one N column: lex (n, j) inside
    # the word -> reshape + permute restores [K, N]
    nib = _nibbles(qw, bits).reshape(qw.shape[0], N, EPF).permute(0, 2, 1)
    nib = nib.reshape(K, N)

    if qs.shape == (G, N):
        s = qs
    elif qs.shape == (N, G):
        s = qs.t().contiguous()
    else:
        raise AssertionError((name, tuple(qs.shape), (G, N)))

    # AutoRound stores g_idx even for desc_act=false as the identity map;
    # only treat it as act-order when it actually permutes the groups.
    if g_idx is not None and not torch.equal(
            g_idx.to(torch.long), torch.arange(K, device=g_idx.device) // wg):
        # act-order: dequantize through the per-position group map, requant.
        # qzeros rows are packed along N like AutoGPTQ ([G, N//EPF] -> [G, N]).
        gmap = g_idx.to(torch.long)
        srow = (s if s.shape == (G, N) else s.t())[gmap]  # [K, N]
        assert kz is not None, f"{name}: desc_act requires qzeros"
        zn = _nibbles(kz, bits).reshape(kz.shape[0], -1)  # [G, N] lex (w, j)
        assert zn.shape == (G, N), (name, tuple(zn.shape), (G, N))
        zrow = zn.to(torch.float32)[gmap] + 1.0           # effective zero
        w = (nib.to(torch.float32) - zrow) * srow
        _sanity(w, name)
        return {"fmt": "w", "w": w.contiguous(),
                "requant": "gptq-actorder", "src": "gptq-actorder"}

    # aligned groups: decode zero row [G, N]
    if kz is not None:
        zn = _nibbles(kz, bits).reshape(kz.shape[0], -1)  # [G, N]
        if zn.shape != (G, N) and zn.shape == (N, G):
            zn = zn.t().contiguous()
        assert zn.shape == (G, N), (name, tuple(zn.shape), (G, N))
        ze = zn.to(torch.float32) + 1.0                   # effective zero [G, N]
    else:
        ze = torch.full((G, N), 8.0, device=nib.device)

    if sym and bool((ze == 8.0).all().item()) and w4_native(plan):
        # symmetric grid: nibble - 8 == our engine decode -> native repack
        packed = _pairs_from(nib)                         # [K//2, N]
        return {"fmt": "nib", "nib_T": packed, "scale_T": s.contiguous(),
                "group": wg, "src": f"gptq-sym-g{wg}"}
    if sym and bool((ze == 8.0).all().item()):
        # symmetric but unsupported group size (e.g. 16): requant at load
        w = (nib.to(torch.float32) - 8.0) * s.repeat_interleave(wg, dim=0)
        _sanity(w, name)
        return {"fmt": "w", "w": w.contiguous(),
                "requant": f"gptq-grid-g{wg}", "src": f"gptq-grid-g{wg}"}
    if sym:
        raise RuntimeError(f"{name}: symmetric GPTQ with unexpected zeros "
                           f"(values {ze.flatten()[:8].tolist()}...)")
    # asymmetric native path (GPTQ zeros need +1 baked into the stored row)
    zt = (ze.to(torch.uint8) if ze.max() <= 255 else None)
    if zt is None:
        raise RuntimeError(f"{name}: zero point overflow")
    return {"fmt": "nibz", "nib_T": _pairs_from(nib),
            "scale_T": s.contiguous(), "zero_T": zt.contiguous(),
            "group": wg, "gptq_plus1": True, "src": f"gptq-asym-g{wg}"}


def decode_awq(raw, plan, name):
    """AutoAWQ decode: qweight int32 [K, N//EPF] (AWQ nibble order), qzeros
    int32 [G, N//EPF] used directly, scales [G, N]."""
    bits = plan["bits"]
    if bits != 4:
        raise NotImplementedError(
            f"{name}: legacy AutoAWQ {bits}-bit is not supported; use an int4 "
            "AWQ / GPTQ / compressed-tensors checkpoint")
    qw = raw["qweight"]                                   # [K, N//EPF]
    kz = raw["qzeros"]
    qs = raw["scale"].to(torch.float32)
    wg = plan["group"]
    EPF = 32 // bits
    nib = _nibbles(qw, bits).reshape(qw.shape[0], -1, EPF)
    nib = _deinterleave(nib, bits).reshape(qw.shape[0], -1)     # [K, N]
    zn = _nibbles(kz, bits).reshape(kz.shape[0], -1, EPF)
    zn = _deinterleave(zn, bits).reshape(kz.shape[0], -1)       # [G, N]
    K, N = nib.shape
    G = K // wg if wg > 0 else 1
    if qs.shape == (G, N):
        s = qs
    elif qs.shape == (N, G):
        s = qs.t().contiguous()
    else:
        raise AssertionError((name, tuple(qs.shape), (G, N)))
    z = zn if zn.shape == (G, N) else zn.t().contiguous()       # real zero
    if wg <= 0 or not w4_native(plan):
        zK = z if wg <= 0 else z.repeat_interleave(wg, dim=0)   # [K, N]
        w = (nib.to(torch.float32) - zK.to(torch.float32)) * \
            (s if wg <= 0 else s.repeat_interleave(wg, dim=0))
        _sanity(w, name)
        return {"fmt": "w", "w": w, "requant": "awq-grid"}
    return {"fmt": "nibz", "nib_T": _pairs_from(nib), "scale_T": s.contiguous(),
            "zero_T": z.to(torch.uint8).contiguous(), "group": wg,
            "src": f"awq-asym-g{wg}"}


# --------------------------------------------------------------------------
# top-level per-projection conversion
# --------------------------------------------------------------------------

def convert_proj(reader, name, plan):
    """Raw tensors of one projection -> engine pieces (see formats above).

    fmt "dense": checkpoint keeps this projection unquantized (ignore list) --
    returns its bf16 [out, in] weight for the caller to quantify.
    """
    if plan["kind"] == "native":
        return None
    if plan["kind"] == "compressed-tensors":
        if not reader.has(f"{name}.weight_packed"):
            assert reader.has(f"{name}.weight"), f"missing tensors for {name}"
            return {"fmt": "dense", "weight": reader.get(f"{name}.weight")}
        raw = {"weight_packed": reader.get(f"{name}.weight_packed"),
               "weight_scale": reader.get(f"{name}.weight_scale"),
               "weight_zero_point": reader.get(f"{name}.weight_zero_point")
               if reader.has(f"{name}.weight_zero_point") else None,
               "weight_shape": reader.get(f"{name}.weight_shape")
               if reader.has(f"{name}.weight_shape") else None}
        return decode_ct(raw, plan, name)
    if plan["kind"] == "gptq":
        if not reader.has(f"{name}.qweight"):
            assert reader.has(f"{name}.weight"), f"missing tensors for {name}"
            return {"fmt": "dense", "weight": reader.get(f"{name}.weight")}
        raw = {"qweight": reader.get(f"{name}.qweight"),
               "scale": reader.get(f"{name}.scales"),
               "qzeros": reader.get(f"{name}.qzeros")
               if reader.has(f"{name}.qzeros") else None,
               "g_idx": reader.get(f"{name}.g_idx")
               if reader.has(f"{name}.g_idx") else None}
        return decode_gptq(raw, plan, name)
    if plan["kind"] == "awq":
        if not reader.has(f"{name}.qweight"):
            assert reader.has(f"{name}.weight"), f"missing tensors for {name}"
            return {"fmt": "dense", "weight": reader.get(f"{name}.weight")}
        raw = {"qweight": reader.get(f"{name}.qweight"),
               "scale": reader.get(f"{name}.scales"),
               "qzeros": reader.get(f"{name}.qzeros")
               if reader.has(f"{name}.qzeros") else None}
        return decode_awq(raw, plan, name)
    raise NotImplementedError(plan["kind"])
