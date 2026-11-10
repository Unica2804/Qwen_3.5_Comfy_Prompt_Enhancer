"""Native PyTorch vision tower for Qwen3.5 (image input path).

Hand-ported from `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionModel`.
The implementation is kept as close to the HF source as is practical so HF
precomputed positions / tables are reusable and the math stays comparable
token-for-token:

* **bf16 throughout** — matches the checkpoint encoding; Conv3d runs bf16/fp32
  kernels natively on this card.
* **Per-image variable-length attention** — images are processed whole
  (`scaled_dot_product_attention` per segment, not flash cu_seqlens), which
  matches eager attention in the HF reference.
* **Patch embedding** is `Conv3d` with kernel [temporal, patch, patch] —
  benchmarked at the same cost as an unfolded linear on this card.
* The merger uses the *pre-shuffle* norm path (norm over 1024, then concat),
  matching `use_postshuffle_norm=False` in HF.
* Weight loading is explicit (`load_vision` / `offload_vision`) so the node
  can keep the tower on pinned CPU while the text engine is running, and
  bring it onto the GPU only for an image encode.
"""
import gc
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden, cu_seqlens, pos_emb):
        """hidden: [total, dim] bf16; pos_emb: (cos[total, hd], sin[total, hd])."""
        total = hidden.shape[0]
        qkv = self.qkv(hidden).reshape(total, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)          # [total, heads, hd]
        cos, sin = pos_emb                                   # [total, hd]
        cos = cos.float().unsqueeze(1)                      # [total, 1, hd]
        sin = sin.float().unsqueeze(1)
        q = (q.float() * cos + _rotate_half(q.float()) * sin).to(hidden.dtype)
        k = (k.float() * cos + _rotate_half(k.float()) * sin).to(hidden.dtype)
        outs = []
        for i in range(1, cu_seqlens.numel()):
            s, e = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
            qh = q[s:e].transpose(0, 1).unsqueeze(0)
            kh = k[s:e].transpose(0, 1).unsqueeze(0)
            vh = v[s:e].transpose(0, 1).unsqueeze(0)
            # eager math exactly as HF: fp32 softmax, cast back to input dtype
            att = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
            att = att.softmax(dim=-1, dtype=torch.float32).to(vh.dtype)
            o = torch.matmul(att, vh)
            # [1, heads, len, hd] -> [len, heads*hd] (HF transposes 1<->2 first)
            outs.append(o.transpose(1, 2).reshape(e - s, self.num_heads * self.head_dim))
        return self.proj(torch.cat(outs, dim=0))


class Qwen3_5VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Qwen3_5VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(self, x, cu_seqlens, pos_emb):
        x = x + self.attn(self.norm1(x), cu_seqlens, pos_emb)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        ts = config.temporal_patch_size
        k = config.patch_size
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                              kernel_size=[ts, k, k], stride=[ts, k, k],
                              bias=True)

    def forward(self, hidden_states):
        # hidden_states: [sum_frames, 3, ts, k, k]  (bf16 input expected)
        return self.proj(hidden_states).flatten(1)


class Qwen3_5VisionPatchMerger(nn.Module):
    """Pre-shuffle merger: LayerNorm over hidden_size, then concat + GELU + out."""

    def __init__(self, config):
        super().__init__()
        self.merge = config.spatial_merge_size
        self.unit = self.merge * self.merge
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(config.hidden_size * self.unit,
                                    config.hidden_size * self.unit)
        self.linear_fc2 = nn.Linear(config.hidden_size * self.unit,
                                    config.out_hidden_size)

    def forward(self, x):
        # x: [sum_tokens, hidden] in patch order (tok ascending); merge 2x2.
        merged = x.view(-1, self.unit, x.shape[-1])
        merged = self.norm(merged)                      # [n, unit, hidden]
        merged = merged.view(merged.shape[0], self.unit * merged.shape[-1])
        merged = self.linear_fc2(F.gelu(self.linear_fc1(merged)))
        return merged


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        return torch.outer(t, self.inv_freq)


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.patch_embed = Qwen3_5VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings,
                                      config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config)

    # -- pre-computed positions -----------------------------------------------

    def fast_pos_embed_interpolate(self, grid_thw):
        device = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        num = self.num_grid_per_side
        out = []
        for t, h, w in grid_thw.tolist():
            h_idxs = torch.linspace(0, num - 1, h, device=device, dtype=torch.float32)
            w_idxs = torch.linspace(0, num - 1, w, device=device, dtype=torch.float32)
            h_f = h_idxs.int(); w_f = w_idxs.int()
            h_c = (h_f + 1).clip(max=num - 1); w_c = (w_f + 1).clip(max=num - 1)
            dh = h_idxs - h_f; dw = w_idxs - w_f
            base_h, base_hc = h_f * num, h_c * num
            idx = [
                (base_h[:, None] + w_f[None, :]).view(-1),
                (base_h[:, None] + w_c[None, :]).view(-1),
                (base_hc[:, None] + w_f[None, :]).view(-1),
                (base_hc[:, None] + w_c[None, :]).view(-1),
            ]
            wt = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).view(-1),
                ((1 - dh)[:, None] * dw[None, :]).view(-1),
                (dh[:, None] * (1 - dw)[None, :]).view(-1),
                (dh[:, None] * dw[None, :]).view(-1),
            ]
            idx_t = torch.stack(idx).to(torch.long)
            wt_t = torch.stack(wt).to(dtype)
            pos = self.pos_embed(idx_t) * wt_t.unsqueeze(-1)
            pos = pos[0] + pos[1] + pos[2] + pos[3]
            merge = self.spatial_merge_size
            pos = pos.repeat(t, 1)
            pos = (pos.view(t, h // merge, merge, w // merge, merge, -1)
                   .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
            out.append(pos)
        return torch.cat(out, dim=0)

    def rot_pos_emb(self, grid_thw):
        merge = self.spatial_merge_size
        max_hw = max(max(h, w) for _, h, w in grid_thw.tolist())
        freq_table = self.rotary_pos_emb(max_hw)              # (max_hw, dim/2)
        device = freq_table.device
        total = sum(t * h * w for t, h, w in grid_thw.tolist())
        pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)
        off = 0
        for t, h, w in grid_thw.tolist():
            mh, mw = h // merge, w // merge
            rows = torch.arange(mh, device=device)
            cols = torch.arange(mw, device=device)
            ir = torch.arange(merge, device=device)
            ic = torch.arange(merge, device=device)
            row = (rows[:, None, None, None] * merge
                   + ir[None, None, :, None]).expand(mh, mw, merge, merge).reshape(-1)
            col = (cols[None, :, None, None] * merge
                   + ic[None, None, None, :]).expand(mh, mw, merge, merge).reshape(-1)
            coords = torch.stack((row, col), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            pos_ids[off: off + coords.shape[0]] = coords
            off += coords.shape[0]
        emb = freq_table[pos_ids]                            # [total, 2, dim/2]
        # HF: `embeddings = embeddings.flatten(1)` then *caller* does
        # `emb = torch.cat((emb, emb), dim=-1)` to get the head_dim table.
        emb = emb.flatten(1)                                 # [total, dim]
        emb = torch.cat((emb, emb), dim=-1)                  # [total, 2*dim]
        return emb.cos(), emb.sin()

    def forward(self, hidden_states, grid_thw):
        # Preprocessed flat patches: [sum_tokens, 3*temporal*patch*patch] with
        # HF's row order (tokens already flattened across frames with temporal
        # replication baked in).  HF's patch_embed reshapes to 5D internally;
        # do the same here.
        if hidden_states.dim() == 2:
            temporal = self.config.temporal_patch_size
            patch = self.config.patch_size
            hidden_states = hidden_states.view(
                -1, self.config.in_channels, temporal, patch, patch)
        hidden_states = self.patch_embed(hidden_states)
        pos = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos
        rotary = self.rot_pos_emb(grid_thw)
        total = hidden_states.shape[0]
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=torch.int32, device=grid_thw.device),
            (grid_thw[:, 1] * grid_thw[:, 2]).repeat_interleave(grid_thw[:, 0])
            .cumsum(dim=0, dtype=torch.int32),
        ])
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens, rotary)
        return self.merger(hidden_states)


# ---------------------------------------------------------------------------
# loading / offload
# ---------------------------------------------------------------------------

_VISION = None
_VISION_META = {}
_VTOWER_META = {
    "residual": False,          # True while vision tensors are resident on GPU
}


def load_vision(model_dir, device="cuda", dtype=torch.bfloat16):
    """Build the vision tower and load `model.visual.*` (+ merger) tensors.

    The weights are loaded **outside** Engine (they're not text weights, and
    `load_weights` doesn't handle them). Returns the model on `device`.
    """
    global _VISION, _VISION_META, _VTOWER_META
    if _VISION is not None and _VISION_META.get("path") == model_dir:
        return _VISION
    if _VISION is not None:
        offload_vision()

    cfg = json.load(open(os.path.join(model_dir, "config.json"),
                         encoding="utf-8"))
    vcfg = type("VCfg", (), cfg.get("vision_config", {}))
    model = Qwen3_5VisionModel(vcfg)

    mod = dict(model.named_modules())
    from .quant_formats import CheckpointReader
    reader = CheckpointReader(model_dir)
    try:
        for k in reader.keys():
            if not k.startswith("model.visual."):
                continue
            name = k[len("model.visual."):]
            for suffix in (".weight", ".bias"):
                if not name.endswith(suffix):
                    continue
                module = name[: -len(suffix)]
                obj = mod.get(module)
                if obj is None:
                    raise KeyError(f"unknown vision module: {module} (key {k})")
                tensor = reader.get(k).to(device=device, dtype=dtype)
                getattr(obj, suffix[1:]).data.copy_(tensor)
    finally:
        # handles hold OS file descriptors and safetensors mmap metadata;
        # release them even if a weight is missing/corrupt
        reader.handles.clear()
    model.to(device=device)
    for name, p in model.named_parameters():
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    model.eval()
    _VISION = model
    _VISION_META = {"path": model_dir, "device": str(device)}
    _VTOWER_META["residual"] = True
    return model


def offload_vision():
    global _VISION, _VISION_META, _VTOWER_META
    if _VISION is None:
        return
    _VISION.cpu()
    torch.cuda.empty_cache()
    _VISION = None
    _VISION_META = {}
    _VTOWER_META["residual"] = False
    gc.collect()


def vision_tower_size(model_dir):
    total = 0
    from safetensors import safe_open
    from glob import glob
    for f in glob(os.path.join(model_dir, "*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k.startswith("model.visual."):
                    t = h.get_slice(k)
                    n = 1
                    for s in t.get_shape():
                        n *= s
                    total += n * 2
    return total
