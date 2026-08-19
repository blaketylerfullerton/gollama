

<div align="center">
<img src="https://i.postimg.cc/kg3gqwB1/Gemini-Generated-Image-bws52zbws52zbws5.png" alt="Go Llama" style="max-width: 600px; width: 100%;">

[![CI](https://github.com/blaketylerfullerton/gollama/actions/workflows/ci.yml/badge.svg)](https://github.com/blaketylerfullerton/gollama/actions/workflows/ci.yml)
[![Go version](https://img.shields.io/github/go-mod/go-version/blaketylerfullerton/gollama)](go.mod)
[![License: MIT](https://img.shields.io/github/license/blaketylerfullerton/gollama)](LICENSE)
</div>

This repo started as a rewrite by (mostly) hand of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), but in Go. It has since moved to the **Qwen3** architecture, and it now runs real pretrained Qwen3-0.6B weights:

```text
prompt: "The capital of France is"
output:  Paris, and
```

This is purely to help me understand inference and transformers better.

Not optimal at all, just for learning. Meant to be a hackable, super simple project for understanding how LLM inference works from first principles.

Inference only — no training. **The engine has no dependencies**; only the terminal UI pulls anything in (bubbletea and lipgloss, isolated under `tools/tui`). Nothing under `engine/` imports either.

## Contents

- [Running it](#running-it)
- [The walkthrough](#the-walkthrough)
- [Inspecting a run](#inspecting-a-run)
- [Architecture](#architecture)
- [Engine internals](#engine-internals)
- [Generating text](#generating-text)
- [Watermarking](#watermarking)
- [Tests](#tests)
- [Speed](#speed)
- [Not implemented yet](#not-implemented-yet)
- [Layout](#layout)
- [Demos](#demos)

## Running it

```bash
go run .
```

You get a welcome screen first — the llama on the left, the machine you're about to run on down the right:

```text
                     ▄▄  ▄▄       ╭───────────────────────────────────────────────────────╮
                     ██  ██       │  this machine                                         │
                    ▀███████▄▄    │  host       Blake's MacBook Air                       │
                    █ o  ████     │  chip       Apple M4                                  │
                    ▀██▄▄▄███     │  cores      10 cores (4P + 6E)                        │
                     █████▀       │  gpu        10-core GPU (unused)                      │
                     █████        │  memory     16.0 GB · 8.4 GB free                     │
                    █████         │  platform   darwin/arm64                              │
          ▄▄▄▄▄▄▄▄▄█████          │  runtime    go1.25.0 · GOMAXPROCS 10                  │
   ▄▄███████████████████          │                                                       │
    ████████████████████          │  weights                                              │
    ████████████████████          │  model      qwen3-0.6b                                │
     ▀██▀▀██▀    ▀██▀▀██▀         │  on disk    1.4 GB                                    │
      ██  ██      ██  ██          │  resident   2.8 GB  (bf16 → f32)                      │
      ██  ██      ██  ██          │                                                       │
      ▀▀  ▀▀      ▀▀  ▀▀          ╰───────────────────────────────────────────────────────╯

 ↑↓ choose · enter select · q quit
```

Hardware detection is in `tools/sysinfo/`: sysctls on macOS, `/proc` on Linux, runtime fields everywhere else. The menu has three entries: "select a model" leads to the picker; "what is GoLlama" leads to a short about page; "past conversations" opens a read-only browser over every chat that's been saved to disk. All three come back here rather than exiting through it.

The picker is where the memory arithmetic that actually matters happens: every model this repo knows about, what it costs on disk versus resident in RAM once bf16 is widened to float32, and a verdict — recommended, fits, or too large — against whatever this machine has free right now:

```text
  GoLlama  choose a model to start with                     Blake's MacBook Air · 16.0 GB ram · 8.4 GB free

 ╭──────────────────────────────────────────────────────────────────────────────────────────────╮
 │  models                                                                                       │
 │                                                                                                │
 │  ▸ Qwen3-0.6B                                              596M    1.4 GB     ready  recommended │
 │    Qwen3-1.7B                                              1.7B    3.8 GB    get it         fits │
 │    Qwen3-4B                                                4.0B    8.2 GB    get it    too large │
 │    Qwen3-8B                                                8.2B   15.3 GB    get it    too large │
 │    tiny random model                                         1M         —  built in         fits │
 ╰──────────────────────────────────────────────────────────────────────────────────────────────╯
 ╭────────────────────────────────────────────────────────────╮ ╭────────────────────────────────╮
 │  Qwen3-0.6B                                                 │ │  MEMORY AFTER LOAD             │
 │  The one this repo is built around...                       │ │  recommendations assume you    │
 │                                                              │ │  keep 4.2 GB free              │
 │  28 layers · 16 q heads over 8 kv heads × 128 dims           │ │                                │
 │  596M parameters · 151936 vocabulary · 40960 max context     │ │  resident         2.8 GB       │
 ╰──────────────────────────────────────────────────────────────╯ ╰────────────────────────────────╯
```

Pressing enter on an installed model opens the third screen: a chat with it. Every turn is wrapped in Qwen3's ChatML markers (`<|im_start|>`/`<|im_end|>`) with a short default system prompt — that formatting is what turns a base model's next-token prediction into an assistant reply, and `chatTemplate` in `chat.go` builds it via `Tokenizer.TokenID` rather than `Encode`, since `Encode` never maps a literal `<|im_start|>` typed inside text to its id, only bytes. Models whose tokenizer has no chat markers at all (the built-in tiny random model) fall back to the old plain-text behavior. Typing and pressing enter streams tokens back as they're generated; a second tab shows, per generated token, what it attended to and what else the model ranked highly — the same instrumentation the inspect screen (below) uses, read out as two short ranked lists next to a live conversation instead of full matrices. Every turn is saved to disk as it happens, and "past conversations" (`h` on the welcome menu) lets you reopen and replay any of them later without reloading a model.

Pressing enter on one marked "get it" instead fetches it: a progress screen (`tools/hf/`) pulls `config.json`, `tokenizer.json`, and the weights straight off HuggingFace — resuming a shard that's already on disk at the right size rather than restarting it — then drops you into chat the moment it lands. Nothing runs in another terminal. A fresh clone with nothing downloaded yet still has the built-in "tiny random model" entry, so every screen works before you fetch any weights. The equivalent by hand, if you'd rather:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir checkpoints/qwen3-0.6b
```

`-prompt` seeds the chat's first message. `-no-splash`, or piping stdout, skips all three screens and falls back to the old fixed walkthrough printed straight to the terminal — for scripting, or a terminal bubbletea can't draw on:

```text
checkpoints/qwen3-0.6b
28 layers · 16 q heads / 8 kv heads x 128 dims · 596M params

prompt  "The capital of France is"
5 tokens  The | _capital | _of | _France | _is

next token
   65.7%  " Paris"
    2.8%  " located"
    2.4%  " the"

output  The capital of France is Paris, and

prefill 2.587s · 3 tokens in 2.831s (944ms/token) · kv cache 224 KB/token
```

## The walkthrough

```bash
go run . -v
```

With `-v` the forward pass narrates itself — shapes, intermediate vectors, per-layer magnitudes, rotary tables and attention weights at every stage. None of it is printed by the model: it goes through the `Tracer` hook (see [Engine internals](docs/engine-internals.md#the-tracer-hook)), and with no tracer attached the hooks are no-ops.

The best bit is the attention grid, which makes causal masking obvious — and on real weights also shows the *attention sink*, where nearly every position dumps a large share of its attention onto token 0:

```text
attention weights — layer 0, head 0 (each row attends across the columns)
                 The _capital      _of  _France      _is
  The          1.000        ·        ·        ·        ·
  _capital     0.872    0.128        ·        ·        ·
  _of          0.382    0.001    0.617        ·        ·
  _France      0.633    0.040    0.310    0.018        ·
  _is          0.452    0.006    0.361    0.005    0.176
  · = masked. Each row sums to 1, and token 0 can only ever attend to itself
```

## Inspecting a run

```bash
go run .
```

Pick a tool from the welcome menu — **Ablation**, Attention, Attribution, or Logit Lens — then a model, same as picking Chat. All four open the same inspect screen, just defaulted to a different tab; it loads the checkpoint once, then you **type a prompt and press enter**. It traces a prefill pass plus one pass per generated token, streaming each into the UI as it completes. Edit and run again with `i` — the model stays loaded.

While you type, it shows the live tokenization, so you can see how the prompt will actually be split before running it:

```text
 > The sky is
 3 tokens: The|_sky|_is
 enter to run · 2 tokens to generate (+/-)
```

Then, after the run:

```text
 "The sky is"  →  " blue,"
 steps  prefill   _blue   ,
```

`n`/`p` step between generated tokens, so you can ask how the model arrived at each one separately.

There's also a replay mode, for looking at a run after the fact or on a machine with no checkpoint:

```bash
go run . -trace run.jsonl   # record while the walkthrough runs
go run . -f run.jsonl       # replay it — no checkpoint, no menu, straight to the inspect screen
```

Both render identical data, because both go through `tools/trace/`: the live collector and the file writer share their event constructors.

Keys: `↑↓` layer, `←→` head, `n`/`p` step between generated tokens, `tab` view, `1`–`5` jump to a view (ablation, attention, attribution, logit lens, stages), `a` ablate the selected head, `g`/`G` first/last layer, `esc` back to the picker, `q` quit.

What the four views actually show — the logit lens finding " Paris" by layer 22, direct logit attribution splitting out which heads and MLPs caused it, the attention sink absorbing 94% of later positions' attention, and the one head ablation that flips the prediction — is in **[docs/inspecting.md](docs/inspecting.md)**, with real output from this model at every step.

## Architecture

Qwen3, which is very close to Llama:

**Grouped-query attention (GQA)** — several query heads share one key/value head, so the KV cache is `NHead/NKVHead` times smaller. `k` and `v` project to `NKVHead*HeadDim` while `q` projects to `NHead*HeadDim`.

**Rotary position embeddings (RoPE)** — positions are encoded by *rotating* q and k rather than adding a position vector. `PrecomputeRotary` builds the cos/sin tables once; `ApplyRotary` splits each vector in half and rotates the pairs `(x[i], x[i+half])`. Signs match HuggingFace's `rotate_half`. `rope_theta` comes from config (Qwen3 uses 1e6, Llama 2 uses 10000).

**RMSNorm** — normalize by root mean square, no mean subtraction, no bias, but *with* a learned per-dimension scale. Used before attention, before the MLP, and once before the LM head.

**QK-norm** — an RMSNorm applied per attention head to q and k, over `HeadDim`. It runs **before** the rotary step, matching Qwen3. Llama doesn't have this at all.

**SwiGLU MLP** — three matrices rather than two: `down(silu(gate(x)) * up(x))`. This is why the intermediate width isn't the classic `4*NEmbed`.

**Causal attention** — scaled dot-product with a causal mask, so position `t` only attends to positions `≤ t`. Rather than building a full `T x T` score matrix and masking the upper triangle with `-inf`, the masked scores are simply never computed. Softmax subtracts the row max before exponentiating for numerical stability.

**Tied embeddings** — the LM head reuses the embedding table. The `tie_word_embeddings` flag wins over the checkpoint's contents here, which is deliberate: Qwen3-0.6B sets the flag *and* still ships an `lm_head.weight` tensor that is byte-for-byte identical to `model.embed_tokens.weight`. Loading it would cost a redundant 155.6M floats (622MB widened to float32), and HuggingFace itself ties the parameters at construction and ignores whatever is stored.

**No biases anywhere.** `Linear` supports them for other architectures, but Qwen3 doesn't use them.

One thing worth calling out: `HeadDim` is read from config, never derived. Qwen3-0.6B has `head_dim=128` while `hidden_size/num_attention_heads` is `1024/16 = 64`. Deriving it corrupts every weight shape downstream, and the symptom is fluent nonsense rather than an error — so `GPTConfig.Validate()` and the loader's shape assertions exist to make that failure loud.

## Engine internals

How weights get loaded from a HuggingFace checkpoint (BF16/F16 widening, sharded `.safetensors`, which `eos_token_id` actually wins), how the tokenizer's hand-written pretokenizer reproduces Qwen3's RE2-incompatible splitting regex, how the KV cache turns decode into `O(T)` work, and the `Tracer` hook every trace and inspect view above is built on — all in **[docs/engine-internals.md](docs/engine-internals.md)**.

## Generating text

```go
out, err := gpt.Generate(ids, model.GenerateOpts{
	MaxTokens:  64,
	SampleOpts: model.SampleOpts{Temperature: 0.8, TopK: 40, TopP: 0.95, Seed: 1},
	OnToken:    func(id int) { fmt.Print(tok.Decode([]int{id})) },
})
```

Sample from the last logit row, append the token, re-run. `OnToken` streams each token as it arrives instead of making you wait for the whole completion.

The sampling filters compose, and they're applied in this order: temperature divides the logits, softmax turns them into probabilities, then top-k and top-p each discard the tail. Whatever survives is renormalized and drawn from. `Temperature: 0` means greedy — take the argmax and ignore every other setting. Each `Sampler` owns its seeded RNG, so a run reproduces exactly regardless of what else is running.

Stop tokens end the run and aren't included in the result. Both `opts.Stop` and the checkpoint's own `eos_token_id` are honored.

The prompt goes through in one **prefill** pass, then each new token is fed in on its own while the KV cache carries the history forward — see [Engine internals](docs/engine-internals.md#the-kv-cache) for how that cache works.

## Watermarking

```bash
go run . -watermark -prompt "The capital of France is"
```

`tools/watermark/` implements a SynthID-Text-style watermark: a second, independent way to draw tokens from the same model that steers generation toward tokens a secret function scores highly, plus a detector that recovers that bias from text alone — no model access required, no change to what the model itself computes.

```text
prompt  "The capital of France is"
SynthID-Text demo — key 0xc0ffeed15c05eed, 4-token context, 4 tournament layers (16-way)

─── plain (ordinary sampling) ───────────────────────────────────────────
 Paris and is home to the Seine River, which is the primary river used for the transportation of goods and commerce. The city's famous landmarks are the Eiffel Tower, Louvre Museum, Eiffel Tower, and the Louvre Museum are all located in Paris. In addition to these landmarks, there are also several other attractions and cultural centers in Paris that contribute to its development. The

─── watermarked (tournament sampling) ───────────────────────────────────
 Paris, and it's approximately 200,000 km². If the population of Paris was 38,000 and the average number of inhabitants per km² is 18, what is the total number of people from all cities in the region?

To answer the problem, you can use the formula:

Total number of people = (Population of Paris + Population

─── detector ────────────────────────────────────────────────────────────
  plain        mean g 0.492   z  -0.52   (81 scored positions)
  watermarked  mean g 0.579   z   4.94   (81 scored positions)

  z above ~4 is the usual line for "almost certainly watermarked" — ordinary text has no reason to land there.
```

Both texts read as ordinary model output — nothing about the watermarked one looks different — but only one of them carries a statistical signature the detector can pick out without ever seeing how it was generated.

The same comparison is a screen on the welcome menu (`Watermark`, alongside Ablation/Attention/Attribution/Logit Lens): type a prompt, and it runs both generations against the loaded checkpoint and shows the detector's readout for each side by side.

How the tournament-sampling bracket and the z-score detector actually work is in **[docs/watermarking.md](docs/watermarking.md)**.

## Tests

```bash
go test ./...
```

`-short` skips the ones that read the 1.5GB checkpoint; all of those also skip cleanly when it's absent, so a fresh clone passes.

**The one that matters most is `TestRealCheckpointPredictsParis`.** It asserts that the real model, given `"The capital of France is"`, ranks `" Paris"` first with more than 40% probability. That single assertion covers the rotary sign convention, the QK-norm ordering, the GQA head mapping, and every weight transpose at once — because all of those failure modes produce a flat distribution over nonsense rather than an error. It's the difference between "the code runs" and "the math is right."

The rest of what's covered, and why — synthetic checkpoints with deliberately mismatched shapes, cached-vs-uncached drift, the causality property, the Tracer copy contract, the import-graph invariants — is in **[docs/testing.md](docs/testing.md)**.

## Speed

Still slow, but no longer quadratic. On an M-series laptop, Qwen3-0.6B generating 8 tokens:

```text
cached:    5.322s     665ms/token
uncached: 31.179s    3.897s/token     ← identical output
speedup:   5.86x
```

The speedup grows with sequence length, since prefill cost is fixed and only the decode steps benefit. Per-operation:

```text
BenchmarkRealForward-10    2    2390730896 ns/op   uncached, 5 tokens
BenchmarkRealDecode-10     2     483599854 ns/op   one cached decode step
```

484ms per token is still ~1.5 tokens/sec, which is bad. `MatMul` (in [linear.go](engine/model/linear.go)) and `Attention.Forward` (in [attention.go](engine/model/attention.go)) already parallelize across output columns and across heads respectively via goroutines — the numbers above are with that in place — so the remaining problem is arithmetic throughput, not missing parallelism: roughly 2.5 GFLOP/s per core, maybe 10× off what a tuned single-threaded loop should manage. Next in order of payoff:

1. **Flat `[]float32` with explicit strides** instead of `[][]float32`, which is a pointer chase per row.
2. **Quantized weights**, so bfloat16 stops being widened to float32 on load.

## Not implemented yet

- **Batching.** One sequence at a time. Continuous batching across concurrent requests is the next architectural step after the tensor layout work — the cache being separate from `GPT` is what leaves room for it.
- **Performance.** See above. Nothing is optimized.
- **Exact numeric verification.** The `" Paris"` test proves the architecture is right in every way that changes the ranking, but a small error — a misplaced epsilon, say — could still shift logits slightly without being caught. A golden-logits comparison against `transformers` would close that gap.

Despite the name, it doesn't run Llama yet — but it's close. Llama needs optional QK-norm (it has none) and RoPE scaling for 3.1+. Everything else is already config-driven.

## Layout

Two top-level folders. `engine/` is the inference itself and depends on nothing outside the standard library. `tools/` is everything that makes a run watchable — and is where every third-party import lives. If you came here to read how a transformer works, `engine/` is the whole thing and you can ignore the rest.

The full annotated file tree — every file in `main.go`/`engine/`/`tools/`, what it does, and why it's split that way — is in **[docs/layout.md](docs/layout.md)**.

## Demos

**Ablation**

https://github.com/user-attachments/assets/b39f6586-f8fb-46f0-8a81-ca73697455c0

**Attention**

https://github.com/user-attachments/assets/2ac27680-323b-4a28-b33e-d2548cd63f3b
