<div align="center">
<img src="https://i.postimg.cc/kg3gqwB1/Gemini-Generated-Image-bws52zbws52zbws5.png" alt="Go Llama" style="max-width: 600px; width: 100%;">
</div>

This repo started as a rewrite by (mostly) hand of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), but in Go. It has since moved to the **Qwen3** architecture, and it now runs real pretrained Qwen3-0.6B weights:

```text
prompt: "The capital of France is"
output:  Paris, and
```

This is purely to help me understand inference and transformers better.

Not optimal at all, just for learning. Meant to be a hackable, super simple project for understanding how LLM inference works from first principles.

Inference only — no training. **The engine has no dependencies**; only the terminal UI pulls anything in (bubbletea and lipgloss, isolated under `tools/` and `cmd/inspect`). Nothing under `engine/` imports either.

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

Pressing enter on an installed model opens the third screen: a chat with it. Typing and pressing enter streams tokens back as they're generated; a second tab shows, per generated token, what it attended to and what else the model ranked highly — the same instrumentation `cmd/inspect` uses, read out as two short ranked lists next to a live conversation instead of full matrices. Every turn is saved to disk as it happens, and "past conversations" from the welcome menu lets you reopen and replay any of them later without reloading a model.

With a checkpoint in `checkpoints/qwen3-0.6b` this is the real 0.6B model; a fresh clone with nothing downloaded yet still has the built-in "tiny random model" entry, so every screen works before you fetch any weights. To get the real ones:

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

With `-v` the forward pass narrates itself — shapes, intermediate vectors, per-layer magnitudes, rotary tables and attention weights at every stage. None of it is printed by the model: it goes through the `Tracer` hook described below, and with no tracer attached the hooks are no-ops.

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
go run ./cmd/inspect
```

An interactive TUI: it loads the checkpoint once, then you **type a prompt and press enter**. It traces a prefill pass plus one pass per generated token, streaming each into the UI as it completes. Edit and run again with `i` — the model stays loaded.

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
go run . -trace run.jsonl           # record while the walkthrough runs
go run ./cmd/inspect -f run.jsonl   # replay it
```

Both render identical data, because both go through `tools/trace/`: the live collector and the file writer share their event constructors.

The **logit lens** is the reason this exists. It projects the residual stream through the LM head at *every* layer, so you can watch an answer get found:

```text
  layer   prediction          prob     rank      H
  0       " only"            18.1%   #11876   3.43  ██████░░░░░░░░░░░░░░░░░░
  4       " not"             54.6%   #12549   2.37  █████████████░░░░░░░░░░░
  7       " a"               54.8%    #3056   2.15  █████████████░░░░░░░░░░░
  12      " a"               20.7%   #10541   3.56  ████░░░░░░░░░░░░░░░░░░░░
  17      "____"             76.6%    #3597   1.05  ██████████████████░░░░░░
  20      "____"             55.9%      #43   1.26  █████████████░░░░░░░░░░░
  21      "____"             45.1%       #9   1.92  ██████████░░░░░░░░░░░░░░
  22      " Paris"           55.1%       #1   1.87  █████████████░░░░░░░░░░░   ← first leads here
  25      " Paris"           96.4%       #1   0.31  ███████████████████████░
  27      " Paris"           65.7%       #1   2.31  ███████████████░░░░░░░░░
  out     " Paris"           65.7%       #1   2.31  ███████████████░░░░░░░░░
```

Early layers guess generic function words. By layer 7 it knows a noun phrase is coming (`" a"`). Around 17 it's reaching for a blank (`"____"`). **" Paris" only takes the lead at layer 22 of 28**, then sharpens to 96% before the last layer hedges back down.

The two extra columns are there because the top row can't tell you the whole story. `rank` is where the eventual answer stood at that depth: it sits past ten thousand for two thirds of the stack, then goes **#43 → #9 → #1** over three layers. Reading the prediction column alone, " Paris" appears from nowhere at 22; reading the rank, it was climbing hard from 19 onwards, and 22 is where a move already underway crosses the line. `H` is the entropy of the whole distribution in nats, which is the model's confidence *independent of which token is winning* — layer 17 is 76.6% sure of `"____"` at H 1.05, and the last layer drops to H 2.31 while still leading with " Paris". That final hedge is a widening of the whole distribution, not a change of mind.

**Direct logit attribution** answers the question none of the above can: not what the model thought at each depth, but which parts of it are *why*. Every component — each attention head, each MLP, the embedding — adds its own vector into the residual stream, and with the final norm's scaling held at what the finished stream produced, the rest is linear. So an output logit is exactly the sum of the components' individual pushes on it, and each push can be reported separately:

```text
  layer 26   pushing " Paris"

  component     Δlogit   ‖write‖
  head 0        +3.700    43.737              │████████████
  head 1        -1.781    32.796        ██████│
  head 7        -0.629    13.032            ██│
  head 9        -0.144    39.767              │
  …
  mlp           +0.335   114.404              │█

  largest across the whole pass: L26 head 0 +3.70, L24 mlp +3.58, L22 mlp +2.90, L23 mlp +2.46
```

`‖write‖` is how much the component moved the residual stream at all; `Δlogit` is how much of that landed on the answer. They come apart constantly, and that gap is the whole point. Head 9 makes the second-largest write in the layer and contributes **nothing** to " Paris" — whatever it's doing, the answer doesn't depend on it. The MLP writes nearly three times as hard as any head and still only adds +0.34. Head 0, writing less than head 9, supplies +3.70 on its own, and head 1 spends most of a comparable write pushing the answer *down*.

An attention pattern tells you a head looked somewhere. Only this tells you the output depended on it — which is why the two views are worth reading side by side: a striking pattern attached to no push is easy to over-read on its own.

Across the whole pass, three of the five largest contributions are MLPs in layers 22–24 — exactly the stretch where the rank column shows " Paris" climbing from #43 to #1. The heads move the stream harder; the MLPs are where the movement becomes the answer.

The attention view shows the sink getting dramatic with depth — at layer 27, token 0 absorbs 94% of every later position's attention:

```text
  layer 27  head 0
                 The _capital      _of  _France      _is
  The          1.000        ·        ·        ·        ·
  _capital     0.983    0.017        ·        ·        ·
  _of          0.962    0.018    0.020        ·        ·
  _France      0.971    0.002    0.009    0.018        ·
  _is          0.862    0.005    0.010    0.027    0.095
  token 0 absorbs 94% of later positions' attention on average
```

Attribution and the attention grid are both still just *watching* the forward pass. **Ablation** is the one view that intervenes: pick a head, force its output to zero before it's merged back into the residual stream, and re-run — if the answer actually moves, the head mattered; if it doesn't, whatever attribution measured wasn't load-bearing. Forcing each of the 448 `(layer, head)` pairs in Qwen3-0.6B to zero, one at a time, on `"The capital of France is"`, exactly one of them flips the top prediction away from `" Paris"` — layer 0, head 3:

```text
  layer   baseline           prob    ablated L0H3       prob
  0       " only"           18.1%    " only"            26.4%
  2       " not"            22.8%    " the"             12.0%
  5       " not"            16.3%    " a"               42.0%
  7       " a"              54.8%    ","                38.1%
  12      " a"              20.7%    ","                28.3%
  17      "____"            76.6%    ","                23.1%
  20      "____"            55.9%    " a"               38.1%
  22      " Paris"          55.1%    " a"               31.7%
  25      " Paris"          96.4%    " a"               19.5%
  27      " Paris"          65.7%    " a"               12.4%
```

The two runs already disagree by layer 2 — long before the baseline commits to " Paris" at layer 22 — so this head is doing something far upstream of where attribution said the decision happens, and the model never recovers " Paris" once it's gone. `cmd/inspect`'s fifth tab (`5`) shows this comparison live: pick a layer and head with the arrow keys, press `a` to ablate it, and see which layer the two runs first disagree at.

Keys: `↑↓` layer, `←→` head, `n`/`p` step between generated tokens, `tab` view, `1`–`5` jump to a view, `a` ablate the selected head, `g`/`G` first/last layer, `q` quit.

Stepping between tokens is where it gets interesting. On `"The capital of France is"` the answer lands at layer 22. On the *next* token — after "…is Paris." — the model predicts `" The"`, and that one settles by layer 19. Different tokens are decided at different depths.

Temperature is easier to believe when you can watch it work on a fact the model actually knows:

```text
  temperature 0.7        temperature 1.0        temperature 1.5
   93.62%  " Paris"       65.69%  " Paris"       15.37%  " Paris"
    1.02%  " located"      2.78%  " located"      1.87%  " located"
    0.80%  " the"          2.35%  " the"          1.67%  " the"
```

## How the printing works

The model doesn't print anything itself. `model.Tracer` is a hook that receives intermediate values as the forward pass runs:

```go
type Tracer interface {
	Stage(layer int, name string, x [][]float32)
	Attention(layer, head int, weights [][]float32)
	Rotary(layer, head int, before, after []float32)
	Note(layer int, format string, args ...any)
}
```

Set `gpt.Trace` and you get the walkthrough. Leave it nil — which is what tests and any real inference do — and every call is a no-op behind a single nil check. That way the pedagogy lives outside the forward pass, and things you can't see from outside a block (attention weights, pre/post-rotary vectors) are still reachable.

**The contract is that implementations must not retain slices past the call.** The engine is free to hand over scratch buffers it means to reuse, so anything you want to keep has to be copied. That's deliberate: buffer reuse is the largest allocation win left in the forward pass, and putting the copy on the tracer costs nothing when tracing is off. Implementations also don't need to be goroutine-safe — every call comes from the goroutine running the pass, and if the matmuls are ever parallelized, emissions will stay on the serial path.

Two extensions hang off it, both opt-in by type assertion, because neither is free. The logit lens costs one extra LM head projection per layer:

```go
type LogitLensTracer interface {
	Tracer
	LogitLens(layer int, logits []float32, target int)
}
```

`target` is the token the pass ended up predicting, which is what makes the rank column possible. It also forces an ordering: the target isn't known until the stack finishes, so the readouts are emitted afterwards in layer order rather than interleaved with the blocks that produced them. Same cost, strictly more information.

Attribution costs one extra `Wo`-sized matmul per layer, to split each head's output back out before `Wo` sums them:

```go
type AttributionTracer interface {
	Tracer
	AttributionTopK() int
	Attribution(layer, component int, tokens []int, effects []float32, norm float64)
}
```

`AttributionTopK` is a second gate on top of the type assertion, so a tracer that only sometimes wants attribution doesn't have to change its type to say so — returning zero turns the whole path off, recording included. The chat UI leaves it off; `-trace` and `cmd/inspect` turn it on.

Effects are reported against the pass's own top candidates rather than the whole vocabulary. Attributing every token would mean a full LM head projection per component — sixteen extra forward passes per layer — and the question worth asking is which components pushed the tokens that were actually in contention. Folding the final norm into the unembedding directions once, instead of into every component, keeps the whole thing to a few hundred thousand multiply-adds against a 155M-parameter LM head.

The engine type-asserts for both, so a tracer that wants neither simply doesn't implement the methods and never pays for them. `trace.Tee` has to be careful here: it advertises an extension only when one of its children would actually use it, which is why there's a type per combination rather than one that implements everything.

**Why the sum is exact, and where the approximation is.** Everything after the last block is a norm and a linear map. RMSNorm's only nonlinearity is the scaling factor it computes from the whole vector — hold that at the value the *finished* stream produced and the norm becomes linear, so the output logit for a token is exactly the sum of the components' effects on it. `RMSNorm.Scale` exists to expose that factor for this. `TestAttributionSumsToTheLogit` checks the identity on both forward paths, and `TestAttributionSumsToTheLogitOnRealWeights` checks it survives the real model, where the stream reaches magnitudes in the hundreds and cancellation has room to bite.

What this measures is the *direct* path: what a component contributed by writing into the stream that reaches the LM head. A layer-2 head that matters only because layer-20 read what it wrote shows up under layer 20, not layer 2. That's the standard meaning of direct logit attribution, and it's a real limitation rather than a rounding error — reading it as "total influence" is the one way to get badly wrong conclusions from a number that always adds up.

### How the pieces are separated

```text
engine/model/  ──Tracer──▶  tools/walkthrough/     terminal walkthrough
                       └─▶  tools/trace/  ┬─ Writer  ──▶ JSONL file ─┐
                                          └─ Collector ─ in memory ──┴─▶ cmd/inspect (TUI)
```

One rule holds it together: **nothing under `engine/` imports `tools/` or `cmd/`.** That's what the two top-level folders are for. The interface lives in `engine/model/` because that's where it's consumed; every implementation lives outside. `TestEngineHasNoThirdPartyDependencies` and `TestEngineDoesNotImportTooling` parse the import graph and fail if either invariant breaks.

`Writer` and `Collector` share their event constructors, so a live view and a replayed file are looking at byte-identical data and the UI needs only one rendering path. Both copy every slice they keep, as the contract requires.

The consequences are what make this worth the indirection: the UI is testable against hand-written fixtures with no checkpoint and no inference, it can be rewritten without touching the engine, and the optimization work can restructure the hot path freely as long as the trace format holds.

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

## Loading real weights

```go
gpt, err := model.FromDirectory("checkpoints/qwen3-0.6b")
```

Reads HuggingFace `config.json` for the architecture and `model.safetensors` for the weights. Handles:

- **F32, BF16 and F16.** BF16 is just float32 with the low mantissa bits chopped off, so widening is a 16-bit shift. F16 needs real exponent rebiasing plus subnormal handling.
- **Sharded checkpoints** via `model.safetensors.index.json`.
- **Shape assertions on every tensor**, with errors accumulated through `errors.Join` so one run reports every mismatch instead of failing on the first.

Stop tokens come from `generation_config.json` when it's present, falling back to `config.json`. They disagree, and it matters: Qwen3-0.6B's `config.json` says `eos_token_id: 151645` while `generation_config.json` says `[151645, 151643]`. HuggingFace generates with the latter, so reading only `config.json` means never stopping on `<|endoftext|>`.

No transposes needed anywhere: PyTorch `nn.Linear` already stores weights `(out, in)` row-major, which is exactly what `Linear.Weight` wants. (GPT-2's `Conv1D` stores them transposed, which is one of several reasons this targets Qwen3 instead.)

## The tokenizer

Byte-level BPE, loaded from a HuggingFace `tokenizer.json`. Three details the format demands:

**Merges come in two shapes.** Older `tokenizer.json` files write them as space-separated strings (`"Ġ Ġ"`), newer ones as explicit pairs (`["Ġ","Ġ"]`). Qwen3 uses the newer form. Both load — the string form can't be parsed unambiguously when a token itself contains a space, which is presumably why they changed it.

**`added_tokens` live outside `model.vocab`,** and their ids continue past it. Qwen3 has 151643 vocab entries (ids 0–151642) plus 26 special tokens at 151643–151668. Sizing the id table from `len(vocab)` drops all of them silently — they'd decode to `""`, so you'd never see `<|im_end|>`.

**`VocabSize()` is not the model's `vocab_size`.** Checkpoints pad the embedding table for alignment: the tokenizer knows 151669 tokens while `config.json` declares 151936. The model can emit logits for ids that decode to nothing. Size embedding tables from config, not from the tokenizer.

**The pretokenizer is hand-written, because RE2 can't do it.** `tokenizer.json` ships its own splitting regex, and Qwen3's uses negative lookahead (`\s+(?!\S)`) — which Go's `regexp` rejects by design, since it's RE2. So `splitQwen` in [pretokenize.go](engine/tokenizer/pretokenize.go) implements that pattern's seven branches directly, including the backtracking the alternation depends on.

`compilePretokenizer` recognises the pattern by string comparison and only claims exactness for the one actually implemented; anything else falls back and `PretokenizerIsExact()` reports `false`.

Two rules do most of the work, and both surprise people:

- **The optional character before a word is any non-letter non-digit**, not just a space. So `f(x)` splits as `f` + `(x` + `)`, and `a\tb` as `a` + `\tb` — the paren and the tab attach to the following word exactly the way a space does.
- **`\s+(?!\S)` gives up its last character.** A run of two spaces before a word splits as `" "` + `" word"`, because the lookahead forces the whitespace branch to backtrack one character so the word branch can claim it. That single rule is why byte-level BPE vocabularies are full of `Ġword` entries.

Verified by cross-checking against the pattern run through Python's `re` (with `\p{L}` and `\p{N}` translated to equivalent classes) over a 34-case corpus — every case matched, including the whole corpus as one blob with tabs, newlines, and unicode.

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

The prompt goes through in one **prefill** pass, then each new token is fed in on its own while the KV cache carries the history forward.

## The KV cache

```go
cache := model.NewKVCache(gpt.Config)
logits, err := gpt.ForwardCached(ids, cache) // appends ids, returns the last row
```

Only keys and values are cached, never queries. A query is used once, by the token that issued it, and then it's done — whereas every future token attends back over every past key and value. That asymmetry is the whole trick.

Two savings. Keys and values for positions already stored aren't recomputed, so a decode step is `O(T)` work rather than `O(T²)`. And `ForwardCached` runs the LM head on one row instead of every row — at 155.6M parameters that's the largest matmul in the model, and only the final position's distribution is ever used.

**The cache is deliberately not a field on `GPT`.** The model is just weights; the cache is per-conversation state. Keeping them separate means one loaded model can serve many independent generations at once. Pass a `Cache` in `GenerateOpts` to continue a previous run rather than reprocessing its history.

**The bug to watch for is the position offset.** A cached token sits at absolute position `cache.Len()`, not 0, so it needs `cos[cache.Len()]`. Get it wrong and the model still runs and still emits fluent text — same silent failure mode as the rotary sign convention. `CausalAttention` takes an explicit `offset` for exactly this reason, and `GenerateOpts.NoCache` exists so the cached path can be diffed against the uncached one.

Memory is `NLayer × NKVHead × HeadDim × 2 × 4` bytes per token. For Qwen3-0.6B that's **224KB per token**, or 9.4GB at the full 40960-token context — and 18.8GB without GQA, since `NKVHead` is half `NHead`. That division is what grouped-query attention buys you.

## Tests

```bash
go test ./...
```

151 tests. `-short` skips the ones that read the 1.5GB checkpoint; all of those also skip cleanly when it's absent, so a fresh clone passes.

**The one that matters most is `TestRealCheckpointPredictsParis`.** It asserts that the real model, given `"The capital of France is"`, ranks `" Paris"` first with more than 40% probability. That single assertion covers the rotary sign convention, the QK-norm ordering, the GQA head mapping, and every weight transpose at once — because all of those failure modes produce a flat distribution over nonsense rather than an error. It's the difference between "the code runs" and "the math is right."

The unit tests build synthetic checkpoints on disk using real HuggingFace tensor names, over a tiny config that deliberately keeps `HeadDim != NEmbed/NHead` and `NKVHead < NHead` — so anything that wrongly derives head dim, or confuses query width with kv width, fails there rather than in a 600MB file.

A few others worth naming:

- `TestCachedMatchesUncachedIncrementally` — feeds tokens one at a time and compares against a full uncached pass over the same prefix at every step. Position drift compounds, so a late-position offset bug surfaces here even when positions 0 and 1 happen to agree. `TestRealCheckpointCachedMatchesUncached` does the same across 28 layers and the real rotary tables.
- `TestForwardIsCausal` — truncating the input must not change the logits for the positions that remain, because no position may attend to the future. The strongest correctness property available without a reference implementation.
- `TestLoaderRejectsWrongHeadDim` — feeds the loader the value `NEmbed/NHead` would have produced and confirms it fails loudly naming `q_proj`.
- The sampling tests check the *distribution*, not just the range: 20000 draws against known probabilities.
- `TestGenerateDoesNotMutatePrompt` — the naive implementation passes this only when the caller's slice has no spare capacity.
- The tokenizer tests run against the real `tokenizer.json` as well as fixtures, to catch format drift in a file I don't control.
- `TestWriterCopiesBeforeReturning` enforces the Tracer contract by scribbling `-999` over every slice right after handing it to the trace writer, then checking none of it reached the file. Without this, buffer reuse in the engine would silently corrupt traces.
- `TestEngineHasNoThirdPartyDependencies` and `TestModelDoesNotImportPresentation` parse the import graph and fail if the engine ever grows a dependency or starts depending on something that consumes it.
- The inspector's views are rendered at four terminal sizes, including 20x5, because off-by-one errors in layout code are invisible until someone resizes.

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
- **Chat templates.** `added_tokens` are loaded and decode correctly, but `Encode` doesn't yet split special tokens out of input text, so `<|im_start|>` in a prompt gets byte-level encoded rather than mapped to its id.
- **Performance.** See above. Nothing is optimized.
- **Exact numeric verification.** The `" Paris"` test proves the architecture is right in every way that changes the ranking, but a small error — a misplaced epsilon, say — could still shift logits slightly without being caught. A golden-logits comparison against `transformers` would close that gap.

Despite the name, it doesn't run Llama yet — but it's close. Llama needs optional QK-norm (it has none) and RoPE scaling for 3.1+. Everything else is already config-driven.

## Layout

Two top-level folders. `engine/` is the inference itself and depends on nothing
outside the standard library. `tools/` is everything that makes a run watchable
— and is where every third-party import lives. If you came here to read how a
transformer works, `engine/` is the whole thing and you can ignore the rest.

```text
main.go              flag parsing, the printed walkthrough (-no-splash / piped),
                     and setup() — loads a checkpoint, or falls back to a
                     random model when one isn't there
chat.go              the interactive path: wires the picker's choice into a
                     live *model.GPT and turns typed lines into streamed tokens
                     plus the per-token attention/candidate summaries the
                     chat screen's inspect tab shows

engine/
  tokenizer/         byte-level BPE, hand-written pretokenizer, tokenizer.json
  model/
    config.go        GPTConfig + HuggingFace config.json loader + validation
    safetensors.go   .safetensors reader, F32/BF16/F16, sharded checkpoints
    loader.go        maps Qwen3 tensor names onto the structs
    trace.go         the Tracer hook the forward pass narrates through
    gpt.go           the model, both forward paths, rotary table management
    kvcache.go       cached keys and values, and what they cost per token
    generate.go      prefill + decode loop, stop tokens, streaming
    sample.go        greedy / temperature / top-k / top-p sampling
    block.go         one transformer layer: pre-norm attention + pre-norm MLP
    attention.go     grouped-query attention, causal attention, softmax
    mlp.go           SwiGLU feed-forward
    norm.go          RMSNorm with a learned scale
    rotary.go        RoPE precompute + apply
    linear.go        Linear layer and matmul
    embedding.go     token id → vector lookup
    ops.go           elementwise helpers (residual add)

tools/
  walkthrough/       the walkthrough Tracer, plus vector/matrix pretty-printers
  trace/             the trace format: events, JSONL writer, in-memory collector,
                     and the tee that fans events out to several consumers
  tui/               the three-screen flow (bubbletea, lipgloss)
    flow.go          Start() — wires welcome → picker → about back into a loop
    welcome.go       screen 1: this machine, and what's on disk to run on it
    picker.go        screen 2: every known model, memory cost against this
                     machine, and a fits/recommended/too-large verdict
    catalog.go       the model list itself — names, architectures, what's
                     installed under a checkpoint root
    chat.go          screen 3: the conversation, plus the inspect tab
    about.go         the "what is GoLlama" page
    history.go       screen: "past conversations" — read-only playback of
                     whatever tools/history has saved
    layout.go        the frame both screens share: header, toolbar, panels
    llama.go         the animated ASCII llama
    wordmark.go      the "GoLlama" title art
    style.go         shared lipgloss styles and the amber color ramp
  history/           persists chat transcripts as JSON, one file per
                     conversation, under ~/.gollama/history
  sysinfo/           what hardware this is about to run on

cmd/inspect/         interactive TUI: type a prompt, run it, inspect the trace

.github/workflows/   CI: go build ./... and go test ./... on every push to main
```
