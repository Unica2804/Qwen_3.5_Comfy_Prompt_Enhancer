<p align="center">

# ✨ Qwen3.5 Prompt Enhancer

**Type a few words. Get a rich, ready-to-use image prompt — expanded by a local AI, right inside ComfyUI.**

</p>

```
  "a cat on a windowsill"  ──►  Qwen3.5 Prompt Enhancer  ──►  "A fluffy ginger cat curled on a sunlit
                                                               windowsill, soft golden-hour light,
                                                               dust motes, shallow depth of field…"
```

No API keys. No accounts. No monthly fees. Everything runs **100% offline on your own GPU**, with no extra packages to install and nothing leaving your machine.

---

## 📋 What you need

| | |
|---|---|
| 🖥️ **ComfyUI** | The embedded/portable build is perfect (tested on the Windows portable release). |
| 🎮 **NVIDIA GPU** | CUDA + Triton. Tested on an **RTX 2050 with 4 GB VRAM** — the 2B model idles at ~1.8 GB. |
| 📦 **A Qwen3.5 checkpoint** | A supported quantized checkpoint (see [Supported models](#-supported-models)). Not included — bring your own. |
| 🍎 macOS / AMD / CPU-only | ❌ Not supported — the engine is built on CUDA Triton kernels. |

> 💡 Everything runs on ComfyUI's own embedded Python. There is nothing to `pip install`.

---

## 🚀 Quick start

**1. Install the node**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Unica2804/Qwen_3.5_Comfy_Prompt_Enahancer.git
```

**2. Add a model**

Copy your Qwen3.5 checkpoint folder into `ComfyUI/models/LLM`. A typical folder looks like:

```
ComfyUI/models/LLM/
└── Qwen3.5-2B/
    ├── config.json
    ├── tokenizer.json
    ├── chat_template.jinja
    └── model-00001-of-00002.safetensors   (one or more .safetensors files)
```

**3. Restart ComfyUI.**

**4. Add the node**

Right-click the canvas → **Add Node → text/Qwen3.5 → Qwen3.5 Prompt Enhancer (INT8)**, then wire its `prompt` output into your text encoder (e.g. CLIP Text Encode).

That's it type an idea, press **Queue**, watch a fully written prompt appear.

---

## 🎛️ Using the node

The node is called **`Qwen3.5 Prompt Enhancer (INT8)`**. Here's every input in plain English:

| Setting | What it does | Good to know |
|---|---|---|
| `text` | Your idea or keyword list, e.g. `cyberpunk city, rain, neon` | The only input you *must* fill in. Keep it short  that's the whole point. |
| `model` | Which checkpoint to use | A dropdown of every supported model it found. |
| `memory_mode` | How VRAM is handled | `offload_after_run` frees VRAM after every run (default). `keep_loaded` keeps the model on the GPU for instant repeat runs. |
| `max_tokens` | Upper limit on the generated prompt length | `512` / `1024` / `2048` / `4096`. Default `1024`. The `structured` and `tag` targets cap at 512. |
| `target` | The output format | See [Picking a target](#-picking-a-target) below. |
| `system_prompt` | The instruction that shapes the writing style | Ships with a sensible default. Edit only if you want a different voice. |
| `temperature` | Creativity | Lower = more predictable, higher = more varied. `0` means always pick the most likely token. |
| `top_p` / `top_k` | How adventurous the sampling is | Defaults `0.9` / `20` work well for prose. |
| `min_p` | Drops unlikely tokens before sampling | Default `0.05`; `0` disables. |
| `repetition_penalty` | Discourages repeating itself | Default `1.05`. |
| `presence_penalty` | Mild penalty for any word already used | Default `0.0` (off). |
| `seed` | Random seed | Change it (or keep the ↻ auto-increment) to reshuffle candidates. |
| `best_of` | Writes N candidates, keeps the best | `1`–`4`. Genuinely cheap — see [Best of N](#-best-of-n). |
| `stream_preview` | Live-typing preview on the node | See [Streaming preview](#-streaming-preview). |
| `image` *(optional)* | A reference picture for the model to look at | See [Image reference](#-image-reference). |

**Outputs**

| Output | What it is |
|---|---|
| `prompt` | The finished, expanded prompt (also shown inline on the node). |
| `token_count` | How many tokens the final prompt uses. |
| `report` | Machine-readable JSON: how many candidates were tried, scores, and any warnings. |

**Recommended starting point:** `target: default`, `best_of: 2`, `memory_mode: offload_after_run`, everything else on its default.

---

## 🎯 Picking a target

Roughly how you want the prompt to look, not a second model:

| Target | Output shape | Best for |
|---|---|---|
| `default` | A flowing, grammatically correct paragraph | FLUX, SD3, Qwen-Image and most modern models |
| `structured` | Four labeled sections, `Subject:` / `Scene:` / `Details:` / `Style:` | Models that respond well to sectioned prompts |
| `tag` | A comma-separated tag list | SDXL-style anime/illustration checkpoints |

Both `structured` and `tag` cap output at **512 tokens** and clean the output for you (markdown stripped, duplicate tags removed).

---

##  Extras worth knowing about

###  Best of N

Set `best_of` to `2`–`4` and the node writes several candidates at once — sharing one prefill, so extra candidates cost almost nothing — then scores them on keyword coverage, budget fit, structure and repetition, and keeps the winner. It's the single most useful knob for getting *all* your keywords into the final prompt.

###  Streaming preview

Set `stream_preview: True` and the partial prompt appears **live on the node** while it's being generated, like watching someone type. Uses the bundled `web/` extension — no setup needed, just make sure the node was added after a full restart.

###  Image reference

Connect an image to the optional `image` input and the model *looks at it* while writing the prompt — ideal for "describe this mood/character/outfit and expand it". Requires a Qwen3.5 checkpoint **with a vision tower** (i.e. `vision_config` in `config.json` and `model.visual.*` weights). Note: `best_of` is disabled in this mode (single candidate).

###  Keyword preservation

Words and phrases from your input are treated as must-keep. If the final output drops one, the `report` (and the ComfyUI log) tells you — raise `best_of` to fix it.

---

##  Supported models

Any **Qwen3.5** checkpoint whose `config.json` declares a `qwen3_5*` model type with `hidden_size` and `layer_types`, plus at least one `.safetensors` file. The 0.8B and 2B tiers are the tested sweet spots.

| Checkpoint format | Works? | Memory footprint (2B) |
|---|---|---|
| Native W8A16 (8-bit) | ✅ | ~1.8 GB |
| Native W4A16 (4-bit, group ≥ 32) | ✅ | ~1.7 GB |
| compressed-tensors (symmetric int4 g32 / int8) | ✅ | ~1.8 / ~2.5 GB |
| GPTQ int4 (e.g. g64 native, g16 → auto-converted) | ✅ | ~0.8–1.0 GB |
| AutoRound / AutoAWQ asymmetric int4 g128 | ✅ | ~0.4 GB |
| Legacy AutoAWQ **8-bit** | ❌ | Unsupported — the loader tells you so |

Folder discovery, in order: `ComfyUI/models/LLM` → paths in the `QWEN35_MODEL_PATHS` environment variable → `EXTRA_MODEL_ROOTS` in `nodes.py` (defaults to `D:\models`, edit the file if yours lives elsewhere). If two discovered folders share a name, the dropdown prefixes the parent folder so you can tell them apart.

---

##  FAQ & troubleshooting

<details>
<summary><b>The dropdown says “(no models found in models/LLM)”</b></summary>

The node couldn't find any checkpoint. Check that:

1. Your folder lives at `ComfyUI/models/LLM/<checkpoint>/` (not directly in `models/`, not in `checkpoints/`).
2. The folder contains `config.json` and at least one `.safetensors`.
3. `config.json` has `"model_type": "qwen3_5..."` with `hidden_size` and `layer_types`.

If your models live somewhere else, set `QWEN35_MODEL_PATHS` to the folder (`;`-separated on Windows, `:` on Linux/macOS) and restart ComfyUI.
</details>

<details>
<summary><b>“Out of memory” / CUDA errors when loading</b></summary>

- Use `memory_mode: offload_after_run` (the default) so a diffusion model can reclaim VRAM after enhancing.
- Close other GPU-hungry apps, or try the 0.8B model — it needs ~1.1 GB.
- The node first asks ComfyUI to free *only the shortfall*, and if the load still fails it retries after a heavier eviction. If you still see the error, your GPU is simply full — try `keep_loaded` off and generate the image *after* enhancing, not while it's resident.
</details>

<details>
<summary><b>The first run is slow — is that normal?</b></summary>

Yes. The checkpoint is read and converted when the engine loads, so the first generation takes a while. Within one ComfyUI session, `offload_after_run` keeps the weights in pinned RAM between runs (much faster than re-reading the file from disk), and `keep_loaded` skips the reload entirely.
</details>

<details>
<summary><b>The result dropped a keyword I typed</b></summary>

Small models like to paraphrase. Set `best_of` to `2` or `4` — the scorer specifically rewards keyword coverage and keeps the candidate that kept your words. Check the `report` output for per-candidate coverage scores and warnings.
</details>

<details>
<summary><b>Does anything leave my computer?</b></summary>

**No.** There is no telemetry, no server, no account. The model runs locally; the output never touches a network.
</details>

<details>
<summary><b>“Legacy AutoAWQ 8-bit checkpoints are not supported”</b></summary>

That checkpoint (`bits: 8`, W8A16 packing) can't be loaded by the engine, and it fails loudly rather than producing garbage. Use the W4/int4 variant of that model, or a native W8A16 build instead.
</details>

<details>
<summary><b>Where do I file a bug?</b></summary>

On the [issue tracker](https://github.com/Unica2804/Qwen_3.5_Comfy_Prompt_Enahancer/issues). Include your GPU, ComfyUI version, the checkpoint + format, and the console log lines starting with `[qwen35]`.
</details>

---

##  Performance at a glance

Measured on an **RTX 2050 4 GB** (ComfyUI embedded Python 3.13, torch 2.11+cu130, Triton 3.7), 256-token prefill, warm:

| | Qwen3.5-0.8B (W8A16) | Qwen3.5-2B (W4A16) |
|---|---|---|
| Memory in use | ~1.1 GB | ~1.8–1.9 GB |
| Decode speed | 13 ms/token (~76 tok/s) | 18 ms/token (~56 tok/s) |
| 200-token prompt | ~2.7 s | ~3.4 s |

For kernel-level benchmarks, parity numbers and the full engine design, see **[ENGINEERING.md](ENGINEERING.md)**.

---

## 🗂️ What's in this repo

```
nodes.py          the ComfyUI node (settings, VRAM handling, streaming preview)
engine.py         the INT8 Triton inference engine
weights.py        format detection + model discovery
kernels/          Triton kernels
targets.py        output formats (default / structured / tag) and scoring
web/              live preview extension
dev/              test suite and benchmarks (not needed to run the node)
ENGINEERING.md    the deep dive: internals, formats, tests, benchmarks
```

## 🤝 Contributing & license

Issues and pull requests are welcome on [GitHub](https://github.com/Unica2804/Qwen_3.5_Comfy_Prompt_Enahancer). Please run the relevant `dev/` tests and describe your GPU + checkpoint when reporting problems.