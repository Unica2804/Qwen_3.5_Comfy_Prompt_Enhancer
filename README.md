# Qwen3.5 Prompt Enhancer (INT8 Triton)

A ComfyUI custom node that turns a short idea into a detailed text-to-image
prompt, using a local **Qwen3.5** checkpoint driven by a hand-written **INT8
Triton** inference engine.

Everything runs on the embedded Python that ships with ComfyUI — no extra
packages, no server, no API keys.

```
STRING (short idea) ──► [Qwen3.5 Prompt Enhancer] ──► STRING (detailed prompt)
```

## Node

Category: `text/Qwen3.5`

| Input | Type | Notes |
|---|---|---|
| `text` | STRING (multiline) | The idea / keyword list to expand. |
| `model` | combo | Any Qwen3.5 checkpoint found in the scanned folders. |
| `memory_mode` | combo | `offload_after_run` (frees VRAM after every run) or `keep_loaded` (resident, faster repeats). |
| `max_tokens` | combo | Generation cap: `512` / `1024` / `2048` / `4096`. |
| `system_prompt` | STRING (multiline) | Defines the enhancer's style. |
| `target` | combo | `default` (prose paragraph), `structured` (Subject/Scene/Details/Style headers), `tag` (comma list) — see [targets](#targets). |
| `temperature`, `top_p`, `top_k` | | Sampling controls. `temperature = 0` is greedy. |
| `min_p` | FLOAT | Drops tokens below this fraction of the top probability (default `0.05`, `0` disables). |
| `repetition_penalty` | FLOAT | Penalty on tokens already generated (default `1.05`, matching the native sampler). |
| `presence_penalty` | FLOAT | Constant penalty for any token that has appeared at least once (default `0.0`). |
| `seed` | | Sampling seed. |
| `best_of` | INT | 1–4 candidates from one shared prefill, best-scoring kept (default `1`). |
| `stream_preview` | BOOLEAN | Push decoded partial text live to the browser (default `False`). |

Outputs: `prompt` (also shown inline on the node), `token_count`,
`report` (JSON: winner, per-candidate scores, warnings).

### Model discovery

Checkpoint folders are collected from, in order:

1. `ComfyUI/models/LLM`
2. every path in the `QWEN35_MODEL_PATHS` environment variable (`;` separated)
3. `EXTRA_MODEL_ROOTS` in `nodes.py` (currently `D:\models`)

A folder qualifies if its `config.json` has a `qwen3_5*` `model_type` with
`hidden_size` and `layer_types`, and it contains at least one `.safetensors`.
The combo label is the folder name, prefixed with the parent folder when two
roots contain the same name. Point `EXTRA_MODEL_ROOTS` at wherever your
checkpoints live.

### Memory behaviour

* `offload_after_run` — weights are copied to **pinned host memory** and VRAM is
  released (`torch.cuda.empty_cache`). A following run reloads from pinned RAM,
  which is far faster than re-reading the safetensors file.
* `keep_loaded` — the engine stays on the GPU between runs.
* Only one engine is kept resident; selecting a different model offloads the
  previous one first. Before a load, the node estimates the engine's actual
  VRAM requirement (from `config.json`: the packed weights, states, KV cache and
  runtime overhead) and asks `comfy.model_management` to free **only the
  shortfall**, so a diffusion model that still fits next to the engine is never
  evicted. If the load still runs out of VRAM, one heavier eviction is attempted
  and the load retried.
* The KV cache is allocated for the real sequence (prompt + `max_new_tokens`)
  and only for the full-attention layers, and never grows past an 8k-position
  context cap. If a prompt does not leave room for `max_tokens` under that cap,
  its first tokens are dropped **with a visible warning** on the node instead of
  silently losing context.
* The 2B engine occupies ~1.8 GB allocated on load (~1.93 GB peak with a full
  400-token prompt in flight), leaving usable headroom on a 4 GB card. Note that
  `keep_loaded` still holds the engine between runs — prefer
  `offload_after_run` when generating images after enhancing.

## Checkpoints

### Native formats (detected from tensor names)

| | **W8A16** | **W4A16 groupwise** |
|---|---|---|
| weights | `<proj>.qweight` int8 `[in, out]` | `<proj>.packed_weight` uint8 `[out, in//2]` |
| weight scale | `<proj>.scale` bf16 `[out]` | `<proj>.scale` bf16 `[out, in//128]` |
| unpack | `w * scale` | `(nibble - 8) * scale`, even K in the low nibble |
| example | Qwen3.5-0.8B | Qwen3.5-2B (`D:\models\qwen-2.5-2B-W4A16`) |

### External formats (detected from `quantization_config.json`)

AWQ (`quant_method: awq`), GPTQ (`quant_method: gptq`) and compressed-tensors
(`quant_format` in the config) checkpoints are converted at load time:

| checkpoint | decodes to | engine path |
|---|---|---|
| CT symmetric int4, group 32 | packed nibbles | **native, bit-exact** (`nib - 8`, direct repack) |
| CT symmetric int8 | dequantised float | per-row int8 requant (rel-err ≈ 0.5 % of ‖w‖) |
| GPTQ symmetric int4, group 64 | packed nibbles | **native** (identity `g_idx`; the engine grid *is* the GPTQ grid) |
| AutoRound (`quant_method: auto-round`, `packing_format: auto_round:auto_gptq`) | same tensors as GPTQ | same as the GPTQ row (e.g. g16 → requant to g32) |
| GPTQ anything else (group 16, act-order, asymmetric) | dequantised float | symmetric int4 requant to group **32** |
| AutoAWQ asymmetric int4 g128 (incl. legacy stacks) | packed nibbles + per-group real zeros | **native** — kernel decodes `nibble − z` per weight group |

Notes:

* The engine always quantises activations dynamically (int8, per-64 group), so
  any weight format feeds the same `tl.dot(int8, int8)` GEMMs.
* Native W4 weight groups: 32, 64, or any multiple of 64 (`tl.dot` needs
  K ≥ 32). Group 32 runs two sub-dots per K-block with one nibble
  re-interleave (`tl.interleave`) — measured 0.23 ms vs 0.14 ms for group-64+
  on a 4096×6144 decode GEMM (M=1, RTX 2050).
* Asymmetric zeros run through the same kernels via a per-group real-zero
  tensor `Z [K//WGRP, N]`; symmetric is the special case `Z = 8`.
* The embedding table and `lm_head` are taken from the checkpoint (they are
  never int-quantised in these formats) and re-quantised to int8 exactly as
  native checkpoints, so external W4 checkpoints cost no extra VRAM beyond
  their projections.
* Legacy AutoAWQ **8-bit** checkpoints (`bits: 8`, W8A16 packing) are **not**
  supported and raise a clear error at load.

Validated end-to-end on real checkpoints (RTX 2050 4 GB, greedy generation of
coherent text on prompt/keyword probes):

| checkpoint | format | weights | engine peak | decode |
|---|---|---|---|---|
| Qwen3.5-2B compressed-tensors int4 g32 symmetric | CT sym int4 | 1.79 GB | 1.86 GB | 19.4 ms/tok |
| Qwen3.5-2B compressed-tensors int8 g32 symmetric | CT sym int8 | 2.40 GB | 2.46 GB | — |
| Qwen3.5-0.8B AutoRound GPTQ int4 g16 sym | GPTQ g16 → requant g32 | 0.79 GB | 0.85 GB | 13.0 ms/tok |
| Qwen2-0.5B-Instruct legacy AutoAWQ asym g128 | loader-only validation (bit-exact decode) | — | — | — |

### Small-tier coverage (3.3)

Shape-driven from `config.json`, so any supported checkpoint in these sizes
loads without new code. Rows marked ✓ are measured above (this card, greedy
coherence); loader-only rows are decode-validated, engine shape-compatible.

| tier | checkpoint | format | weights | status |
|---|---|---|---|---|
| 0.5B | Qwen2-0.5B-Instruct | legacy AutoAWQ asym int4 g128 | ~0.36 GB | ✓ decode-validated (bit-exact), loader-only e2e |
| 0.8B | Qwen3.5-0.8B | native W8A16 | 1.01 GB | ✓ e2e + parity (97.9 % top-1 vs bf16) |
| 0.8B | Qwen3.5-0.8B | AutoRound GPTQ int4 g16 | 0.79 GB | ✓ e2e, requant to g32 (72.3 % top-1) |
| 2B | Qwen3.5-2B | native W4A16 g128 | 1.73 GB | ✓ e2e + parity (cos 0.998) |
| 2B | Qwen3.5-2B | compressed-tensors int4 g32 | 1.79 GB | ✓ e2e (native, bit-exact) |
| 2B | Qwen3.5-2B | compressed-tensors int8 g32 | 2.40 GB | ✓ e2e (per-row int8 requant) |

### Conversion details

Both formats keep the weights in their checkpoint encoding and quantise
**activations** dynamically, so every weight matmul is
`tl.dot(int8, int8) → int32` on native int8 tensor cores; only the epilogue is
float. The W4 path never materialises a dequantised weight — except on the
requant path listed above, where the checkpoint's own grid cannot be preserved.

## Engine

### Quality engine

The engine's decode loop is ours, so it spends the owned inference stack on
output quality — the three things the native `Load CLIP` + `Generate Text`
path structurally does not do:

**System-prefix KV reuse (2.1).** The system prompt is identical on every
call, so after the first prefill the engine snapshots
`(pos, kv, kvs, dstate, cstate)` at the end of the prefix (on CPU, so
`offload_after_run` keeps its full VRAM benefit) and restores it per call
instead of re-running those tokens. Saves ~29 % of call time on a 151-token
prefix, bit-exact: the split is rounded down to the delta-rule kernel's
32-token chunk tiling (`PREFIX_ALIGN`), because unaligned splits re-bracket
the fp32 recurrence and diverge with depth (measured: 31 % linear-state
divergence at layer 17 for a 57-token split). `test_prefix.py` proves the
reuse equals a full prefill bit-for-bit and that a changed system prompt
recomputes.

**Batched best-of-N (2.2).** `Engine.generate_batch` prefills once, replicates
the state to N physical rows, and decodes all candidates in one pass — rows
see the weight reads once, and because decode is dispatch-bound, extra rows
cost almost nothing. Kernel deltas are row offsets in `delta_decode`,
`conv_silu`, `conv_state_update`, `attn_decode`, `qk_norm_rope`; the single-
stream paths are unchanged (B=1 gives identical kernels). Candidates are
scored deterministically (keyword coverage 40 %, budget fit 25 %, header
structure 20 %, repetition 15 % — `targets.score_candidates`) and the winner
is returned with the per-candidate report. Verified: greedy batch rows are
bit-identical to single-stream greedy; reproducible seeds; and on the 0.8B,
**best-of-4 costs 3.29 s vs 3.54 s for a single 200-token draw** — four
candidates for less than one long draw.

**Header mask (2.3).** `sampler.sample_logits` accepts a `[vocab]` boolean
mask; disallowed logits go to `-inf` before penalties/temperature, so greedy,
top-k and top-p all respect it. The engine constrains line-start positions
(the first token, and any token after a newline) to the allowed header set
plus stop tokens — v1 constrains only the line's first token, letting bodies
generate freely. Prevents 2B drift into prose at line starts.

**Section grammar (2.3 v2).** `SectionGrammar` in `targets.py` constrains the
whole header instead of just its first token: the allowed set per step is
built offline from the tokenizer's canonical tokenisation of every header
plus the colon, the stop tokens are always allowed, and a newline returns to
line-start. After the colon the body is free until the next newline. The mask
per token is a cache lookup — one `[vocab]` boolean tensor per state. Since
the shipped header sets tokenise to uniform length (`['Subject', ':']`,
`['Scene', ':']`, …) the state is unambiguous; non-uniform sets fall back to
the v1 header mask with a log line. While testing it I also found (and fixed)
that `ChatTokenizer.token_to_id("\n")` returned `None` — the vocab token is
the byte-mapped `Ċ` — which had silently disabled the v1 newline tracking
until then.

```powershell
& $py test_grammar.py    # grammar states: first-token, colon, body, newline restart
```

**Targets (2.4).** `targets.py` compiles for the downstream consumer:

| target | system | budget | headers | post-processing |
|---|---|---|---|---|
| `default` | node's `system_prompt` | — | — | — |
| `structured` | 4-section template | 512 | Subject/Scene/Details/Style | strip markdown |
| `tag` | tag-list template | 512 | — | tag dedupe |

Generation runs against the target's budget, headers are constrained by the
mask, and out comes `(prompt, token_count, report)` — values the next node
can act on.

### Activation quantisation

One scale per **64-element group** along the reduction axis (`GROUP = 64`,
matching the GEMM `BLOCK_K`). Each int8 dot is dequantised with its own scale and
accumulated in fp32.

Per-token (single-scale) quantisation was the original design, but a single
outlier then crushes the precision of the other 63 values in the row. On the
0.8B that cost a full point of top-1 agreement. Group-wise scales are strictly
better *and* faster (each `tl.dot` no longer has a loop-carried int32
dependency):

| metric (teacher-forced, 94 positions) | per-token scale | per-64-group scale |
|---|---|---|
| top-1 agreement vs reference | 89.4 % | **94.7 %** |
| mean logit cosine | 0.9896 | **0.9982** |
| mean KL(ref‖engine) | 0.0422 | **0.0093** |
| ref top-1 inside engine top-5 | — | **100 %** |
| decode | 23.1 ms/token | **13.5 ms/token** |

### Embedding-table and lm-head encodings

The tied table used to live twice in VRAM — bf16 for the gather, int8 for the
logits GEMM (1.53 GB on the 2B). Both halves are now quantised, and the choice
of encoding was driven by teacher-forced parity on the 0.8B (95 positions,
HF dequantised reference):

| encoding | top-1 agreement | mean logit cosine |
|---|---|---|
| embed bf16 + head int8 (old) | 94.7 % | 0.9982 |
| **embed int8 + head int8 (shipped)** | **97.9 %** | 0.9975 |
| embed bf16 + head W4 | 86.2 % | 0.9948 |
| embed int8 + head W4 | 86.2 % | 0.9941 |

Per-row int8 on the embedding performed identically or better on every metric
the decoder actually consumes (top-1 / KL), so the 508 MB saving is free. A
W4-packed lm-head — symmetric `[-8, 7]` per-64-group grid, fed through the
same `gemm_w4` kernels — is implemented (`head_w4=True`) but off by default:
4-bit rows are too coarse to reconstruct logit differences, and the ~255 MB
saving does not buy back nine points of top-1.

### W4A16: int4 weights, int8 matmul

The 4-bit checkpoint is kept packed in VRAM — 685 MB for the 2B instead of
1.37 GB if the nibbles were expanded to int8, which matters on a 4 GB card. The
nibbles are expanded to int8 **inside** the dot loop, so the matmul still runs on
the int8 tensor cores and the checkpoint's 4-bit grid is preserved exactly: this
path adds no rounding beyond the checkpoint's own.

Two details make it fast:

* **K-major layout.** At load time `packed_weight [N, K//2]` is transposed to
  `[K//2, N]` and the group scales `[N, K//WG]` to `[K//WG, N]`, so one K-block
  of a tile is a fully coalesced run along N. Storing them the checkpoint way and
  transposing in registers cost 40 % of the achievable bandwidth.
* **No shuffles in the inner loop (fast path).** For native groups ≥ 64 the low
  nibble *is* the even-K weight, so the activation's even/odd split is free
  (`tl.reshape` + `tl.split`) and the block is summed with two plain int8 dots.
  Splitting the K sum in two changes nothing numerically: int32 accumulation is
  exact and both halves share the same activation scale and weight scale group.
  Group-32 checkpoints (compressed-tensors, and requantised GPTQ g16) run a
  second path: one interleave (`tl.trans` + `tl.interleave`) re-assembles the
  even/odd nibbles into a contiguous int8 tile for a single K=32 dot — the
  sub-block called out in the external-format section above.

Measured effect on the decode GEMMs (RTX 2050, ~105 GB/s achievable):

| kernel | before | after |
|---|---|---|
| `mlp_gate_up_quant_w4` | 285 µs | 144 µs |
| `proj3_w4` | 154 µs | 87 µs |
| `gemm_w4` | 114 µs | 46 µs |

### KV cache

The attention KV cache is bf16 by default (the lossless path). An int8 variant
exists behind `Engine(kv_int8=True)` / `QWEN35_KV_INT8=1`: cache values are
int8 with one fp32 scale per (position, head, side) — the K vector and V
vector of each head carry their own `amax/127`, which keeps softmax-score
precision close to bf16 while halving the cache size and the decode KV-read
bandwidth (201 MB → ~101 MB at 4096 positions on the 2B; ~13 MB at 1024).

It is **not** the default: measured against the bf16 path on the parity
suites it costs real fidelity — 0.8B teacher-forced top-1 97.9 % → 94.7 %
(95 positions), 2B mean logit cosine 0.9983 → 0.9975, min 0.9833 → 0.9749,
argmax still 7/7. The ~100 MB at long context is not worth a visible quality
move on a 4 GB card, so bf16 stays the default and int8 is there for the day
long context matters more than the last cosine point. `dev/test_kv_int8.py`
covers the kernel-level agreement between the two encodings.

### Fused kernels

| kernel | fuses |
|---|---|
| `rmsnorm_quant` | RMSNorm + group quantisation |
| `proj3` / `proj3_w4` | three projections sharing one activation tile (q/k/v, or qkv+z) |
| `mlp_gate_up_quant` / `_w4` | gate + up GEMMs, SiLU·mul, and quantisation of the down-proj input |
| `norm_gated_quant` | gated RMSNorm + quantisation of the out-proj input |
| `gate_mul_quant` | attention output gate + quantisation |
| `qk_norm_rope` | q/k RMSNorm + partial RoPE + KV-cache write |
| `delta_prefill` / `delta_decode` | gated delta-rule (linear attention) |
| `gemm_i8` / `gemm_w4` | int8 GEMM with optional fused residual add |

The W4 kernels are selected by dtype (`uint8` weights → W4), so `engine.py`
calls the same functions for both formats.

### Layout notes

* Weights are stacked per role with the layer index on dim 0
  (`lin_qkv[18]`, `mlp_gate[24]`, …). W8 `qweight` is already stored
  `[in_features, out_features]` — exactly the GEMM layout; W4 tensors are
  transposed to K-major as described above.
* `embed_tokens` is quantised to int8 with one scale per vocabulary row and is
  dequantised only for the gathered rows (`[T, H]`), saving ~508 MB on the 2B
  with no measured quality cost (teacher-forced top-1 actually improved):
  the gather still reads `[V, H]` rows contiguously.
* `lm_head` is `[hidden, vocab]` int8 with a per-vocabulary-row scale,
  halving logits read bandwidth. A W4-packed head (reusing the `gemm_w4`
  kernels) is implemented behind `head_w4=True` but is **not** the default:
  4-bit rows proved too coarse for logit reconstruction (teacher-forced top-1
  fell from ~95 % to ~86 % on the 0.8B parity set), so the ~255 MB it would
  save is not worth the quality loss. The gather and the GEMM deliberately
  use two tensors — the gather wants `[V, H]`, the GEMM wants `[H, V]`, and
  transposed quantised loads are exactly the bandwidth loss the GEMM kernels
  were written to avoid.
* Attention QK/PV products stay bf16/fp32 — only weight matmuls are int8.
* Everything shape-dependent comes from `config.json` (head counts, head dim,
  rope partial factor, hidden/intermediate size), so the same engine runs the
  0.8B (1024/3584) and the 2B (2048/6144).
* KV cache `[full-attn layers, seq, 2·kv_heads·head_dim]` bf16 — allocated for
  the real sequence budget, not the position cap, and only for the layers that
  actually use it (all-layers allocation wasted 3/4 of it on the 2B). Delta
  state `[18, 16, 128, 128]` fp32, conv state `[18, 3, 6144]` bf16.

## Measured performance

RTX 2050 4 GB (sm_86, 16 SMs, ~105 GB/s), ComfyUI embedded Python 3.13,
torch 2.11+cu130, triton 3.7. 256-token prefill, warm.

| | 0.8B W8A16 | 2B W4A16 | 2B CT int4 (external) |
|---|---|---|---|
| weights | 1.01 GB | 1.73 GB | 1.79 GB |
| VRAM (engine loaded) | 1.07 GB | 1.80 GB | 1.86 GB |
| prefill | 581 tok/s | 548 tok/s | 488 tok/s |
| decode | 13.1 ms/token (76 tok/s) | 17.9 ms/token (56 tok/s) | 19.4 ms/token (52 tok/s) |
| 200-token prompt, warm | ~2.7 s | ~3.4 s | — |
| 200-token prompt, cold | ~4–5 s | ~5–6 s | — |
| peak VRAM (256-token prefill + decode) | 1.11 GB | 1.93 GB | 2.00 GB |

The 2B costs almost nothing extra: it reads fewer weight bytes per token
(685 MB of packed nibbles + 509 MB of int8 `lm_head`, versus 1.26 GB for the
0.8B) and decode is dispatch-bound anyway. The external-format 2B on the
group-32 interleave path is ~8 % slower per token than the native-group path
(the cost of reassembling nibble order) — inside the normal run-to-run
variance at this scale, and far below the cost of shipping a second kernel.

Compared with the pre-footprint engine, the 2B dropped from 2.24 GB to 1.73 GB
of weights and from 2.77 GB to 1.80 GB allocated at load: the embedding table
went from bf16 `embed` + int8 `lm_head` (1.53 GB for one tied table) to int8 on
both sides (1.02 GB), and the KV cache stopped being allocated for all 24
layers at 4096 positions (201 MB) — it now holds only the 6 full-attention
layers, sized to the actual sequence.

### Why decode is dispatch-bound, not bandwidth-bound

One 2B decode step is ~255 Triton launches: CPU enqueue time dominates the
17.9 ms/token budget, and of the GPU work ~4.7 ms is the `lm_head` read alone.
Decode throughput is therefore set by launch count (~56 µs each, Python 3.13
on Windows), not by memory bandwidth. Consequences:

* The workspace view is cached as a plain `SimpleNamespace` (a lazy
  `__getattr__` cost ~4 µs × 896 accesses/token).
* Quantisation passes were fused into their producing kernels, removing 42
  launches.
* Decode tiles stay narrow (`BLOCK_N = 64`) so the 16 SMs get several blocks
  each.

The next step for a large win would be CUDA-graph capture of the decode step
(the per-token position argument is currently a Python int, so this needs the
position passed as a device tensor) or a single full-model decode megakernel.

## Files

```
engine.py        Engine: weights, workspace, prefill/decode/generate, batched
                 best-of-N (generate_batch), prefix KV reuse, header mask
weights.py       safetensors -> stacked tensors, format detection, model discovery
quant_formats.py external-format detection + conversion (CT/GPTQ/AWQ), requant helpers, CheckpointReader
tokenizer.py     chat tokenisation via the checkpoint's own Jinja template
sampler.py       greedy / temperature / top-k / top-p / mask, batched sampling
targets.py       target table (system/budget/headers/post), scoring
nodes.py         the ComfyUI node, model cache, VRAM handling, streaming preview
kernels/         Triton kernels (gemm, w4, norm, attention, linear_attn, common)
web/             live streaming-preview canvas extension
dev/             parity tests, benchmarks, profilers (not needed at runtime)
```

## Tests

Run with the embedded interpreter from `dev/`:

```powershell
$py = "D:\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded\python.exe"
& $py test_kernels.py      # per-kernel parity vs torch reference (W8 path)
& $py test_fused.py        # fused kernels vs unfused composition (bit-exact)
& $py test_w4.py           # W4 kernels vs torch reference (packed int4)
& $py test_w4_grid.py      # W4 kernels: native groups 32/64/128 x sym/asym, proj3 zeros
& $py test_quant.py        # external-format decoders: synthetic round-trips + real-checkpoint requant noise
& $py test_legacy_awq.py   # real legacy AutoAWQ (asym g128) decode vs independent reference
& $py test_kv_int8.py      # int8 KV vs bf16 KV: same inputs, same outputs to quantization noise
& $py test_prefix.py       # system-prefix KV reuse: bit-exact vs full prefill, time saved
& $py test_mask.py         # header mask: sampler unit checks + engine line-start behaviour
& $py test_batch.py        # best-of-N: diverse reproducible rows, greedy == single-stream bit-exact
& $py eval_prompts.py [target] [model]  # 50-prompt structure validity + keyword coverage
& $py verify_w4_load.py    # loaded tensors bit-identical to the checkpoint
& $py test_sampler.py      # sampler knobs: min_p / penalties / top-k (CPU)
& $py test_chunk_prefill.py# 700-token prompt: chunked == unchunked prefill
& $py test_quality.py ref  # dump HF reference logits (once)
& $py test_quality.py check# teacher-forced agreement / cosine / KL
& $py test_ext_e2e.py [tag]# Engine generate smoke on external CT/GPTQ checkpoints
& $py test_ext_quality.py <dir> # teacher-forced quality of an external 0.8B checkpoint vs cached bf16 ref
& $py parity_w4.py         # 2B: per-position + per-decode-step parity vs HF
& $py bench.py [dir]       # prefill + decode throughput (default: 0.8B native)
& $py bench_both.py        # prefill + decode throughput, both native models
& $py bench_e2b.py [dir]   # per-kernel profile of one decode step
& $py test_node.py         # node end-to-end, both models, both memory modes
& $py test_api.py          # same, through a live ComfyUI /prompt API
```

All kernels match the torch reference to bf16 rounding (`rel_l2 ≈ 1.7e-3`);
`rmsnorm_quant`, `gate_proj`, `conv_state_update` and the delta-rule state are
bit-exact.

Chunked prefill now verifies bit-identical (700-token prompt: `rel_l2 = 0`
between `prefill_chunk = 512` and a single-chunk prefill, enabled by the
absolute key positions after the 0.1 fix).

The 2B W4 engine was checked against an HF `Qwen3_5ForCausalLM` built by
dequantising the same checkpoint on the CPU: mean logit cosine **0.9983** over
all 42 prompt positions, cosine ≥ 0.9992 on 8 teacher-forced decode steps, and
the **same argmax token at every step**. The 0.8B sits at **97.9 %** teacher-
forced top-1 agreement (above the 94.7 % before the embedding-table work).

External formats, teacher-forced against the bf16 0.8B reference over 95
positions (`test_ext_quality.py`):

| checkpoint | path | top-1 agreement | mean logit cosine | mean KL | ref top-1 in top-5 |
|---|---|---|---|---|---|
| Qwen3.5-0.8B GPTQ int4 g16 | requant to g32 | 72.3 % | 0.9455 | 0.393 | 95.7 % |

The GPTQ-g16 requant is the honest cost of the format: the checkpoint's own
g16 scales cannot round-trip through a g32 grid, so that path agrees with bf16
less than the native 0.8B (97.9 %). No such cost exists on the bit-exact
native paths (CT int4, GPTQ g64, AutoAWQ g128), which is exactly the requant-
vs-native decision rule `quant_formats.py` implements: keep the checkpoint's
grid whenever the kernels can represent it, requantise only when they cannot.

### Phase 2.5 acceptance (`eval_prompts.py`, 50 fixed ideas)

| model | target | structure validity | keyword coverage |
|---|---|---|---|
| 0.8B (native W8A16) | `structured` (mask on) | **100 %** | **96 %** |
| 0.8B (native W8A16) | `default` | 100 % (no headers required) | 90 % |
| 2B (native W4A16) | `structured` (mask on) | **100 %** | 80 % |

Structure validity (all four headers present and in order) clears the ≥ 99 %
bar on both models with the mask on. Keyword coverage clears ≥ 95 % on the
0.8B but not the 2B (80 %) — the W4 model paraphrases more freely. That is the
honest limitation the mask cannot fix; best-of-N is the tool for it, and the
coverage weight in the scorer is there precisely to select for it.

Batched best-of-4 timing on the same prompt class (0.8B, warm engine,
200-token cap): **best-of-1 3.54 s, best-of-4 3.29 s** — four candidates for
less than one long draw, well inside the ≤ 1.5× bar. This is the dispatch-
bound behaviour: the extra candidate rows share every weight read.

### Phase 4 status

* **Streaming preview** — shipped (server side): `stream_preview=True` pushes
  decoded partial text through `PromptServer.send_sync` (sid-less broadcast)
  every 8 tokens; `web/qwen35_preview.js` renders it on the node canvas.
* **CUDA graphs for decode** — deferred: the position argument is currently a
  Python int (not a device tensor), and shapes vary with `max_new_tokens`; the
  plan flags this as a "large lift, only worth it if the engine stays the
  primary backend". It stays a known item, not attempted in this pass, since
  batch-N + prefix reuse already exploit the dispatch-bound cost structure
  without graph capture.
