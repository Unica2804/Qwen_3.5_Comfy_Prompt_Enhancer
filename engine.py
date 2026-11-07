"""Qwen3.5 INT8 inference engine (prefill + decode) driving the Triton kernels.

The same engine runs both supported checkpoint formats: W8A16 (int8 weights) and
W4A16 groupwise (int4 weights kept packed, expanded inside the dot loop).  Every
shape comes from `config.json`, so a 0.8B and a 2B model use identical code."""
import math
import os
import threading
import time
import types

import torch

from .kernels.attention import attn_decode, attn_prefill, qk_norm_rope
from .kernels.common import GROUP
from .kernels.gemm import gemm_i8, mlp_gate_up_quant, proj3
from .kernels.linear_attn import (conv_silu, conv_state_update, delta_decode,
                                  delta_prefill)
from .kernels.norm import (gate_mul_quant, gate_proj, norm_gated_quant,
                           rmsnorm_quant)
from .weights import load_weights


def rope_inv_freq(rope_theta, dim, device):
    return 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))


class Workspace:
    """Scratch buffers sized for `cap` tokens."""

    def __init__(self, cfg, cap, device):
        self.cap = cap
        self._views = {}
        H = cfg["hidden_size"]
        I = cfg["intermediate_size"]
        NH = cfg["linear_num_value_heads"]
        DK = cfg["linear_key_head_dim"]
        DV = cfg["linear_value_head_dim"]
        conv_dim = 2 * NH * DK + NH * DV
        NQ = cfg["num_attention_heads"]
        HD = cfg["head_dim"]
        V = cfg["vocab_size"]
        d = dict(device=device)
        ng = lambda k: k // GROUP
        self.hidden = torch.empty((cap, H), dtype=torch.bfloat16, **d)
        self.xq = torch.empty((cap, H), dtype=torch.int8, **d)
        self.xs = torch.empty((cap, ng(H)), dtype=torch.float32, **d)
        self.proj = torch.empty((cap, max(conv_dim + NH * DV, NQ * HD * 2 + 2 * 2 * HD)),
                                dtype=torch.bfloat16, **d)
        self.mixed = torch.empty((cap, conv_dim), dtype=torch.bfloat16, **d)
        self.dout = torch.empty((cap, NH * DV), dtype=torch.bfloat16, **d)
        self.aq_o = torch.empty((cap, NQ * HD), dtype=torch.int8, **d)
        self.as_o = torch.empty((cap, ng(NQ * HD)), dtype=torch.float32, **d)
        self.aq_d = torch.empty((cap, I), dtype=torch.int8, **d)
        self.as_d = torch.empty((cap, ng(I)), dtype=torch.float32, **d)
        self.q = torch.empty((cap, NQ * HD), dtype=torch.bfloat16, **d)
        self.aout = torch.empty((cap, NQ * HD), dtype=torch.bfloat16, **d)
        self.hq = torch.empty((cap, I), dtype=torch.int8, **d)
        self.hs = torch.empty((cap, ng(I)), dtype=torch.float32, **d)
        self.beta = torch.empty((cap, NH), dtype=torch.float32, **d)
        self.gate = torch.empty((cap, NH), dtype=torch.float32, **d)
        self.logits = torch.empty((1, V), dtype=torch.bfloat16, **d)
        self.xq1 = torch.empty((1, H), dtype=torch.int8, **d)
        self.xs1 = torch.empty((1, ng(H)), dtype=torch.float32, **d)
        # int8 embedding table gather + per-row dequant scales (1.1)
        self.eq = torch.empty((cap, H), dtype=torch.int8, **d)

    def view(self, T):
        """Cached namespace of the first T rows.

        Attribute access on the result must be a plain slot lookup: decode calls
        this hundreds of times per token, so a lazy __getattr__ is far too slow.
        """
        v = self._views.get(T)
        if v is None:
            v = types.SimpleNamespace()
            for name, val in vars(self).items():
                if (isinstance(val, torch.Tensor) and val.dim() >= 1
                        and val.shape[0] == self.cap and name != "logits"):
                    val = val[:T]
                setattr(v, name, val)
            self._views[T] = v
        return v


class Engine:
    # KV cache starts small and grows toward the context cap on generate (1.3).
    KV_INIT = 1024
    # Hard context budget. A prompt that does not leave room for `max_new_tokens`
    # under this cap loses its first tokens *with a warning* instead of growing
    # the KV cache without bound (an 8k cap is ~100 MB of KV on the 2B).
    KV_CAP = 8192

    def __init__(self, model_dir, device="cuda", max_seq=None, prefill_chunk=512,
                 embed_bf16=False, head_w4=False, kv_int8=None, log=print):
        self.log = log
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.ctx_cap = max_seq if max_seq is not None else self.KV_CAP
        self.max_seq = min(self.ctx_cap, self.KV_INIT)
        self.prefill_chunk = prefill_chunk
        self.embed_bf16 = embed_bf16
        self.head_w4 = head_w4
        # KV cache encoding: int8 (+ per-head fp32 scales) or bf16.  The bf16
        # path is the lossless original; the int8 path halves cache size and
        # KV read bandwidth.  Selection: explicit arg, else the
        # QWEN35_KV_INT8 env var ("1"/"0"), else the default below.
        if kv_int8 is None:
            kv_int8 = os.environ.get("QWEN35_KV_INT8", "0") == "1"
        self.kv_int8 = kv_int8
        self.w = load_weights(model_dir, device=self.device, log=log,
                              embed_bf16=embed_bf16, head_w4=head_w4)
        cfg = self.w.cfg
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.I = cfg["intermediate_size"]
        self.NL = cfg["num_hidden_layers"]
        self.layer_types = cfg["layer_types"]
        self.n_lin = sum(1 for x in self.layer_types if x == "linear_attention")
        self.NH = cfg["linear_num_value_heads"]
        self.DK = cfg["linear_key_head_dim"]
        self.DV = cfg["linear_value_head_dim"]
        self.VD = self.NH * self.DV
        self.conv_dim = 2 * self.NH * self.DK + self.VD
        self.NQ = cfg["num_attention_heads"]
        self.NKV = cfg["num_key_value_heads"]
        self.HD = cfg["head_dim"]
        self.eps = cfg.get("rms_norm_eps", 1e-6)
        self.V = cfg["vocab_size"]
        rope = cfg.get("rope_parameters", {})
        self.RD = int(self.HD * rope.get("partial_rotary_factor", 0.25))
        self.invf = rope_inv_freq(rope.get("rope_theta", 1e7), self.RD, self.device)
        # Only full-attention layers have a KV cache; linear layers carry state in
        # dstate/cstate. Allocating one slot per layer wasted NL - n_full slots.
        self.n_full = sum(1 for x in self.layer_types if x == "full_attention")

        kv_dim = 2 * self.NKV * self.HD          # K and V together, per position
        self.kv = torch.zeros((self.n_full, self.max_seq, kv_dim),
                              dtype=torch.int8 if self.kv_int8 else torch.bfloat16,
                              device=self.device)
        self.kvs = (torch.zeros((self.n_full, self.max_seq, 2 * self.NKV),
                                dtype=torch.float32, device=self.device)
                    if self.kv_int8 else None)
        self.dstate = torch.zeros((self.n_lin, self.NH, self.DK, self.DV),
                                  dtype=torch.float32, device=self.device)
        self.cstate = torch.zeros((self.n_lin, 3, self.conv_dim),
                                  dtype=torch.bfloat16, device=self.device)
        self.ws = Workspace(cfg, prefill_chunk, self.device)
        self.pos = 0
        self._mega = None
        self._collect = False
        self._prefix_snap = None       # (prefix_ids tuple, cpu state) or None
        self.debug = None
        # One generation at a time: pos/kv/dstate/cstate are single mutable state
        # and queued prompts would otherwise corrupt each other.
        self._gen_lock = threading.Lock()
        self.log(f"[qwen35] engine ready on {self.device}, "
                 f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # ------------------------------------------------------------------ state
    def ensure_seq(self, n):
        """Grow the KV cache so `n` positions fit (state is preserved).

        Growth stops at the context cap; `generate` truncates, with a warning,
        against `ctx_cap` instead of letting the cache expand without bound.
        """
        n = min(int(n), self.ctx_cap)
        if n <= self.max_seq:
            return
        kv = torch.zeros((self.n_full, n, 2 * self.NKV * self.HD),
                         dtype=self.kv.dtype, device=self.kv.device)
        if self.pos > 0:
            kv[:, : self.pos] = self.kv[:, : self.pos]
        self.kv = kv
        if self.kvs is not None:
            kvs = torch.zeros((self.n_full, n, 2 * self.NKV),
                              dtype=torch.float32, device=self.kv.device)
            if self.pos > 0:
                kvs[:, : self.pos] = self.kvs[:, : self.pos]
            self.kvs = kvs
        self.max_seq = n

    def reset(self):
        self.pos = 0
        self.dstate.zero_()
        self.cstate.zero_()

    def offload(self):
        self.w.offload()
        self.dstate = self.dstate.to("cpu")
        self.cstate = self.cstate.to("cpu")
        self.kv = self.kv.to("cpu")
        if self.kvs is not None:
            self.kvs = self.kvs.to("cpu")
        self.ws = None
        torch.cuda.empty_cache()

    def reload(self):
        self.w.to(self.device)
        d = self.device
        self.kv = self.kv.to(d)
        if self.kvs is not None:
            self.kvs = self.kvs.to(d)
        self.dstate = self.dstate.to(d)
        self.cstate = self.cstate.to(d)
        self.ws = Workspace(self.cfg, self.prefill_chunk, d)

    @property
    def on_gpu(self):
        return self.w.device.type == "cuda"

    # --------------------------------------------------------------- forward
    def _linear_layer(self, i, li, T, pos0):
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["in_norm"][i], ws.xq, ws.xs, self.eps)
        proj3(ws.xq, ws.xs,
              w["lin_qkv"][li], w["lin_qkv_s"][li],
              w["lin_z"][li], w["lin_z_s"][li], None, None,
              ws.proj, block_k=64,
              z1=w["lin_qkv_z"][li] if w.has_z else None,
              z2=w["lin_z_z"][li] if w.has_z else None)
        gate_proj(ws.xq, ws.xs, w["lin_b"][li], w["lin_a"][li],
                  ws.beta, ws.gate, w["lin_dt"][li], w["lin_alog"][li])
        mixed_in = ws.proj[:, : self.conv_dim]
        conv_silu(mixed_in, w["lin_conv"][li], self.cstate[li], ws.mixed, T)
        conv_state_update(mixed_in, self.cstate[li], T)
        if T == 1:
            delta_decode(ws.mixed[0], ws.gate[0], ws.beta[0], self.dstate[li], ws.dout[0])
        else:
            delta_prefill(ws.mixed, ws.gate, ws.beta, self.dstate[li], ws.dout, T)
        norm_gated_quant(ws.dout, ws.proj[:, self.conv_dim:], w["lin_norm_w"][li],
                         ws.aq_o, ws.as_o, self.eps)
        gemm_i8(ws.aq_o, ws.as_o, w["lin_out"][li], w["lin_out_s"][li],
                ws.hidden, residual=ws.hidden,
                b_z=w["lin_out_z"][li] if w.has_z else None)

    def _attn_layer(self, i, fi, T, pos0):
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["in_norm"][i], ws.xq, ws.xs, self.eps)
        proj3(ws.xq, ws.xs,
              w["attn_q"][fi], w["attn_q_s"][fi],
              w["attn_k"][fi], w["attn_k_s"][fi],
              w["attn_v"][fi], w["attn_v_s"][fi],
              ws.proj, block_k=64,
              z1=w["attn_q_z"][fi] if w.has_z else None,
              z2=w["attn_k_z"][fi] if w.has_z else None,
              z3=w["attn_v_z"][fi] if w.has_z else None)
        kvs = self.kvs[fi] if self.kvs is not None else None
        qk_norm_rope(ws.proj, w["attn_qn"][fi], w["attn_kn"][fi], self.invf,
                     ws.q, self.kv[fi], kvs, pos0, self.eps,
                     kv_int8=self.kv_int8)
        if T == 1:
            attn_decode(ws.q[0], self.kv[fi], ws.aout[0], pos0,
                        kvs=kvs, kv_int8=self.kv_int8,
                        nq=self.NQ, nkv=self.NKV, hd=self.HD)
        else:
            attn_prefill(ws.q, self.kv[fi], ws.aout, T, pos0,
                         kvs=kvs, kv_int8=self.kv_int8,
                         nq=self.NQ, nkv=self.NKV, hd=self.HD)
        gate_mul_quant(ws.aout, ws.proj, ws.aq_o, ws.as_o, hd=self.HD)
        gemm_i8(ws.aq_o, ws.as_o, w["attn_o"][fi], w["attn_o_s"][fi],
                ws.hidden, residual=ws.hidden,
                b_z=w["attn_o_z"][fi] if w.has_z else None)

    def _mlp(self, i, T):
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["post_norm"][i], ws.xq, ws.xs, self.eps)
        mlp_gate_up_quant(ws.xq, ws.xs, w["mlp_gate"][i], w["mlp_gate_s"][i],
                          w["mlp_up"][i], w["mlp_up_s"][i], ws.hq, ws.hs,
                          zg=w["mlp_gate_z"][i] if w.has_z else None,
                          zu=w["mlp_up_z"][i] if w.has_z else None)
        gemm_i8(ws.hq, ws.hs, w["mlp_down"][i], w["mlp_down_s"][i],
                ws.hidden, residual=ws.hidden,
                b_z=w["mlp_down_z"][i] if w.has_z else None)

    # ---- batched decode (2.2) -------------------------------------------
    # The rows of the workspace are B parallel candidates sharing one absolute
    # position.  State tensors carry a leading row axis with the LAYER axis
    # first, so per-layer slices are contiguous views (reshape must be
    # zero-copy, or kernel writes would land in a throwaway buffer):
    #   kv_b    [n_full, B, seq, kv_dim]
    #   dstate  [n_lin, B, NH, DK, DV]      (delta rule, per candidate)
    #   cstate  [n_lin, B, 3, conv_dim]     (conv windows, per candidate)
    # All position-independent kernels see B independent rows, as in prefill.

    def _linear_layer_b(self, i, li, B, pos0):
        w, ws = self.w, self.ws.view(B)
        rmsnorm_quant(ws.hidden, w["in_norm"][i], ws.xq, ws.xs, self.eps)
        proj3(ws.xq, ws.xs,
              w["lin_qkv"][li], w["lin_qkv_s"][li],
              w["lin_z"][li], w["lin_z_s"][li], None, None,
              ws.proj, block_k=64,
              z1=w["lin_qkv_z"][li] if w.has_z else None,
              z2=w["lin_z_z"][li] if w.has_z else None)
        gate_proj(ws.xq, ws.xs, w["lin_b"][li], w["lin_a"][li],
                  ws.beta, ws.gate, w["lin_dt"][li], w["lin_alog"][li])
        mixed_in = ws.proj[:, : self.conv_dim]
        cst = self.cstate_b[li].reshape(B * 3, self.conv_dim)
        conv_silu(mixed_in, w["lin_conv"][li], cst, ws.mixed, B, batch=True)
        conv_state_update(mixed_in, cst, B, batch=B)
        dst = self.dstate_b[li].reshape(B * self.NH, self.DK, self.DV)
        delta_decode(ws.mixed, ws.gate, ws.beta, dst, ws.dout, batch=B)
        norm_gated_quant(ws.dout, ws.proj[:, self.conv_dim:], w["lin_norm_w"][li],
                         ws.aq_o, ws.as_o, self.eps)
        gemm_i8(ws.aq_o, ws.as_o, w["lin_out"][li], w["lin_out_s"][li],
                ws.hidden, residual=ws.hidden,
                b_z=w["lin_out_z"][li] if w.has_z else None)

    def _attn_layer_b(self, i, fi, B, pos0):
        w, ws = self.w, self.ws.view(B)
        rmsnorm_quant(ws.hidden, w["in_norm"][i], ws.xq, ws.xs, self.eps)
        proj3(ws.xq, ws.xs,
              w["attn_q"][fi], w["attn_q_s"][fi],
              w["attn_k"][fi], w["attn_k_s"][fi],
              w["attn_v"][fi], w["attn_v_s"][fi],
              ws.proj, block_k=64,
              z1=w["attn_q_z"][fi] if w.has_z else None,
              z2=w["attn_k_z"][fi] if w.has_z else None,
              z3=w["attn_v_z"][fi] if w.has_z else None)
        seq = self.kv_b.shape[2]
        kv_flat = self.kv_b[fi].reshape(B * seq, 2 * self.NKV * self.HD)
        kvs_flat = (self.kvs_b[fi].reshape(B * seq, 2 * self.NKV)
                    if self.kvs_b is not None else None)
        qk_norm_rope(ws.proj, w["attn_qn"][fi], w["attn_kn"][fi], self.invf,
                     ws.q, kv_flat, kvs_flat, pos0, self.eps,
                     kv_int8=self.kv_int8, batch=True,
                     kv_stride=(seq * 2 * self.NKV * self.HD) if B > 1 else None,
                     kvs_stride=(seq * 2 * self.NKV) if B > 1 else None)
        attn_decode(ws.q, kv_flat, ws.aout, pos0,
                    kvs=kvs_flat, kv_int8=self.kv_int8, batch=B,
                    nq=self.NQ, nkv=self.NKV, hd=self.HD)
        gate_mul_quant(ws.aout, ws.proj, ws.aq_o, ws.as_o, hd=self.HD)
        gemm_i8(ws.aq_o, ws.as_o, w["attn_o"][fi], w["attn_o_s"][fi],
                ws.hidden, residual=ws.hidden,
                b_z=w["attn_o_z"][fi] if w.has_z else None)

    def forward_batch(self, token_ids, pos0=0):
        """One batched decode step: `token_ids` is a list of B ids (one per
        candidate), all at the shared absolute position `pos0`.

        Returns the [B, V] logits for the next token.
        """
        B = len(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        ws = self.ws.view(B)
        emb = self.w["embed"]
        if emb.dtype == torch.int8:
            torch.index_select(emb, 0, ids, out=ws.eq[:B])
            sc = self.w["embed_s"][ids].to(torch.bfloat16).unsqueeze(1)
            torch.mul(ws.eq[:B], sc, out=ws.hidden)
        else:
            torch.index_select(emb, 0, ids, out=ws.hidden)
        lin_i = full_i = 0
        for i in range(self.NL):
            if self.layer_types[i] == "linear_attention":
                self._linear_layer_b(i, lin_i, B, pos0)
                lin_i += 1
            else:
                self._attn_layer_b(i, full_i, B, pos0)
                full_i += 1
            self._mlp(i, B)
        self.pos = pos0 + 1
        w = self.w
        # the workspace [1, H] single-row buffers (xq1/xs1/logits) cannot serve
        # rows >= 1; redo the head with the regular row buffers instead
        rmsnorm_quant(ws.hidden, w["final_norm"], ws.xq, ws.xs, self.eps)
        out = torch.empty((B, self.V), dtype=torch.bfloat16, device=self.device)
        gemm_i8(ws.xq, ws.xs, w["lm_head"], w["lm_head_s"], out,
                block_n=512, block_k=64)
        return out

    # ------------------------------------------------------------ generation
    def generate_batch(self, prompt_ids, n=4, max_new_tokens=256, temperature=0.0,
                       top_k=0, top_p=1.0, seed=None, stop_ids=(248046, 248044),
                       on_token=None, min_p=0.0, repetition_penalty=1.0,
                       presence_penalty=0.0, prefix_len=0, header_mask=None,
                       newline_id=None, line_grammar=None):
        """Batched best-of-N: one shared prefill, then `n` parallel decode rows.

        Rows that hit a stop id (or the token budget) are frozen on token 0.
        Returns a list of `n` token-id lists (independent candidates).  Scoring
        and decoding to text happen in the caller, which owns the tokenizer
        (2.2).  `seed` is the base seed: candidate i samples with seed + i + 1,
        so the batch is fully reproducible.
        """
        from .sampler import sample_logits, sample_batch
        B = max(1, min(int(n), 8))
        with self._gen_lock:
            budget = self.ctx_cap - 1
            if len(prompt_ids) + max_new_tokens > budget:
                prompt_ids = prompt_ids[-(budget - max_new_tokens):]
                prefix_len = 0
            prefix_len = max(0, min(int(prefix_len), len(prompt_ids) - 1))
            prefix_len -= prefix_len % self.PREFIX_ALIGN
            self.ensure_seq(len(prompt_ids) + max_new_tokens + 1)
            if prefix_len > 0:
                prefix = tuple(prompt_ids[:prefix_len])
                rest = prompt_ids[prefix_len:]
                snap = self._prefix_snap
                if snap is not None and snap[0] == prefix:
                    self._restore(snap[1])
                    logits = self._forward_chunks(rest, self.pos)
                else:
                    self.reset()
                    self._forward_chunks(prompt_ids[:prefix_len], 0)
                    self._prefix_snap = (prefix, self._snap())
                    logits = self._forward_chunks(rest, self.pos)
            else:
                logits = self.prefill(prompt_ids)
            # replicate the shared state into per-candidate rows
            kv = self.kv.detach()
            nf, seq, dim = kv.shape
            self.kv_b = (kv.unsqueeze(1).expand(nf, B, seq, dim).contiguous())
            dl, dh, dk, dv = self.dstate.shape
            self.dstate_b = (self.dstate.unsqueeze(1)
                             .expand(dl, B, dh, dk, dv).contiguous())
            cl, c3, cc = self.cstate.shape[0], self.cstate.shape[1], self.cstate.shape[2]
            self.cstate_b = (self.cstate.unsqueeze(1)
                             .expand(cl, B, c3, cc).contiguous())
            self.kvs_b = (self.kvs.unsqueeze(1)
                          .expand(self.kvs.shape[0], B, self.kvs.shape[1],
                                  self.kvs.shape[2]).contiguous()
                          if self.kvs is not None else None)
            out = [[] for _ in range(B)]
            alive = [True] * B
            line_start = [True] * B
            gstates = ([line_grammar.start_state() for _ in range(B)]
                       if line_grammar is not None else None)
            cur = logits.unsqueeze(0).expand(B, -1).contiguous()
            for step in range(max_new_tokens):
                toks = sample_batch(cur, temperature, top_k, top_p, seed, step,
                                    min_p=min_p,
                                    repetition_penalty=repetition_penalty,
                                    presence_penalty=presence_penalty,
                                    out=out, alive=alive,
                                    header_mask=header_mask,
                                    line_start=line_start,
                                    grammar=line_grammar, gstates=gstates)
                done = True
                for b, tok in enumerate(toks):
                    if not alive[b]:
                        continue
                    if tok in stop_ids:
                        alive[b] = False
                        continue
                    out[b].append(tok)
                    if on_token is not None:
                        on_token(tok)
                    line_start[b] = newline_id is not None and tok == newline_id
                    done = False
                if done or self.pos >= self.max_seq - 1:
                    break
                cur = self.forward_batch(toks, pos0=self.pos)
            self.kv_b = None
            self.dstate_b = None
            self.cstate_b = None
            self.kvs_b = None
            return out

    def _final_logits(self, T):
        w, ws = self.w, self.ws.view(T)
        last = ws.hidden[T - 1: T]
        rmsnorm_quant(last, w["final_norm"], ws.xq1, ws.xs1, self.eps)
        gemm_i8(ws.xq1, ws.xs1, w["lm_head"], w["lm_head_s"], ws.logits,
                block_n=512, block_k=64)
        return ws.logits[0]

    def logits_all(self, T):
        """Logits for every one of the T rows currently in the workspace (debug/parity)."""
        w, ws = self.w, self.ws.view(T)
        rmsnorm_quant(ws.hidden, w["final_norm"], ws.xq, ws.xs, self.eps)
        out = torch.empty((T, self.V), dtype=torch.bfloat16, device=self.device)
        gemm_i8(ws.xq, ws.xs, w["lm_head"], w["lm_head_s"], out,
                block_n=512, block_k=64)
        return out

    def forward(self, token_ids, pos0=0):
        """Run `token_ids` (list[int]) starting at absolute position `pos0`; returns logits."""
        T = len(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        ws = self.ws.view(T)
        emb = self.w["embed"]
        if emb.dtype == torch.int8:
            # int8 embedding table: dequantise only the gathered rows (1.1)
            torch.index_select(emb, 0, ids, out=ws.eq[:T])
            sc = self.w["embed_s"][ids].to(torch.bfloat16).unsqueeze(1)
            torch.mul(ws.eq[:T], sc, out=ws.hidden)
        else:
            torch.index_select(emb, 0, ids, out=ws.hidden)
        lin_i = full_i = 0
        self.debug = [] if self._collect else None
        for i in range(self.NL):
            if self.layer_types[i] == "linear_attention":
                self._linear_layer(i, lin_i, T, pos0)
                lin_i += 1
            else:
                self._attn_layer(i, full_i, T, pos0)
                full_i += 1
            self._mlp(i, T)
            if self.debug is not None:
                self.debug.append(ws.hidden[T - 1].float().cpu())
        self.pos = pos0 + T
        return self._final_logits(T)

    # ------------------------------------------------------------ generation
    def _forward_chunks(self, token_ids, pos0):
        """Run ids (chunked) starting at absolute position `pos0`; current logits."""
        logits = None
        for c0 in range(0, len(token_ids), self.prefill_chunk):
            logits = self.forward(token_ids[c0: c0 + self.prefill_chunk],
                                  pos0=pos0 + c0)
        return logits

    def prefill(self, token_ids):
        """Run the prompt (chunked) and return the logits for the next token."""
        self.reset()
        return self._forward_chunks(token_ids, 0)

    # ---- system-prefix KV reuse (2.1) -------------------------------------
    # The system prompt is identical on every call; after the first prefill we
    # snapshot (pos, kv, dstate, cstate) at the end of the prefix on CPU (so
    # offload_after_run keeps its full VRAM benefit) and restore it instead of
    # re-running those tokens.  Keyed by the exact prefix token ids; invalid
    # when the model changes (new Engine).  Saving is exact: the token ids
    # processed and their positions are unchanged, so this equals full prefill
    # bit-for-bit the same way chunked prefill does.
    #
    # One constraint: the gated delta rule re-bases its fp32 state every
    # CHUNK (32) tokens inside `delta_prefill`, so the exactness holds only
    # if the split lands on the same tiling as full prefill — i.e. aligned to
    # multiples of the delta chunk.  An unaligned split brackets the same
    # recurrence differently and diverges through the depth (measured: 31 %
    # linear-state difference at layer 17 for a 57-token split).  Splits are
    # therefore rounded down to the alignment; the remainder (< 32 tokens)
    # re-runs with the user turn, which costs little.
    PREFIX_ALIGN = 32

    def _snap(self):
        """State after the current position, on CPU (safe to keep offloaded)."""
        p = self.pos
        return (p,
                self.kv[:, :p].clone().cpu(),
                self.kvs[:, :p].clone().cpu() if self.kvs is not None else None,
                self.dstate.clone().cpu(),
                self.cstate.clone().cpu())

    def _restore(self, snap):
        p, kv, kvs, dstate, cstate = snap
        d = self.device
        self.kv[:, :p].copy_(kv.to(d))
        if self.kvs is not None and kvs is not None:
            self.kvs[:, :p].copy_(kvs.to(d))
        self.dstate.copy_(dstate.to(d))
        self.cstate.copy_(cstate.to(d))
        self.pos = p

    def decode_step(self, token_id):
        return self.forward([token_id], pos0=self.pos)

    def set_prefill_chunk(self, n):
        """Resize the prefill chunking (recreates the workspace; state is reset)."""
        n = max(1, int(n))
        if n == self.prefill_chunk:
            return
        self.prefill_chunk = n
        self.reset()
        self.ws = Workspace(self.cfg, n, self.device)

    # ------------------------------------------------------------ generation
    def generate(self, prompt_ids, max_new_tokens=256, temperature=0.0, top_k=0, top_p=1.0,
                 seed=None, stop_ids=(248046, 248044), on_token=None,
                 min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0,
                 prefix_len=0, header_mask=None, newline_id=None, line_grammar=None):
        """Sample `max_new_tokens` for one prompt; returns (token_ids, warnings).

        The engine holds a generation lock for the whole call: pos/kv/dstate/
        cstate are single mutable state, so queued runs serialise here instead of
        interleaving and corrupting each other (0.3).

        With `prefix_len > 0` the first `prefix_len` token ids are treated as an
        identical system prefix: the first call snapshots the state after them
        and later calls restore it instead of re-running those tokens (2.1).

        With `header_mask` set (a `[vocab]` boolean tensor), line-start
        positions — the first generated token and every token after a
        `newline_id` — are constrained to the allowed header tokens (2.3 v1).

        With `line_grammar` set (a SectionGrammar), the whole header is
        constrained token by token (2.3 v2): supersedes `header_mask`.
        """
        from .sampler import sample_logits
        with self._gen_lock:
            budget = self.ctx_cap - 1
            warnings = []
            if len(prompt_ids) + max_new_tokens > budget:
                keep = max(1, budget - max_new_tokens)
                dropped = len(prompt_ids) - keep
                prompt_ids = prompt_ids[-keep:]
                if dropped:
                    # 0.2: never fail a workflow, but never silently drop context
                    warnings.append(
                        f"prompt (+{max_new_tokens} new tokens) exceeded the "
                        f"{self.ctx_cap}-token context: the first {dropped} prompt "
                        "tokens were dropped and the result may be degraded")
                    # Truncation removed the start of the prompt, so the
                    # system prefix is no longer leading the sequence.
                    prefix_len = 0
            prefix_len = max(0, min(int(prefix_len), len(prompt_ids) - 1))
            prefix_len -= prefix_len % self.PREFIX_ALIGN  # exact split tiling
            self.ensure_seq(len(prompt_ids) + max_new_tokens + 1)
            gen = None
            if temperature > 1e-5:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(seed if seed is not None else torch.seed()))
            if prefix_len > 0:
                prefix = tuple(prompt_ids[:prefix_len])
                rest = prompt_ids[prefix_len:]
                snap = self._prefix_snap
                if snap is not None and snap[0] == prefix:
                    self._restore(snap[1])
                    logits = self._forward_chunks(rest, self.pos)
                else:
                    self.reset()
                    self._forward_chunks(prompt_ids[:prefix_len], 0)
                    self._prefix_snap = (prefix, self._snap())
                    logits = self._forward_chunks(rest, self.pos)
            else:
                logits = self.prefill(prompt_ids)
            out = []
            line_start = True          # the first generated token starts a line
            gs = line_grammar.start_state() if line_grammar is not None else None
            for _ in range(max_new_tokens):
                if line_grammar is not None:
                    m = line_grammar.mask(gs)
                elif header_mask is not None and line_start:
                    m = header_mask
                else:
                    m = None
                tok = sample_logits(logits, temperature, top_k, top_p, gen,
                                    min_p=min_p,
                                    repetition_penalty=repetition_penalty,
                                    presence_penalty=presence_penalty,
                                    generated=out, mask=m)
                if tok in stop_ids:
                    break
                if line_grammar is not None:
                    gs = line_grammar.advance(gs, tok)
                out.append(tok)
                if on_token is not None:
                    on_token(tok)
                if self.pos >= self.max_seq - 1:
                    break
                line_start = newline_id is not None and tok == newline_id
                logits = self.decode_step(tok)
            return out, warnings
