# Engine internals

How weights actually get loaded, how the tokenizer works, the KV cache, and the `Tracer` hook everything above is built on. The [README](../README.md) covers what to run; this is for reading the code.

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

**The pretokenizer is hand-written, because RE2 can't do it.** `tokenizer.json` ships its own splitting regex, and Qwen3's uses negative lookahead (`\s+(?!\S)`) — which Go's `regexp` rejects by design, since it's RE2. So `splitQwen` in [pretokenize.go](../engine/tokenizer/pretokenize.go) implements that pattern's seven branches directly, including the backtracking the alternation depends on.

`compilePretokenizer` recognises the pattern by string comparison and only claims exactness for the one actually implemented; anything else falls back and `PretokenizerIsExact()` reports `false`.

Two rules do most of the work, and both surprise people:

- **The optional character before a word is any non-letter non-digit**, not just a space. So `f(x)` splits as `f` + `(x` + `)`, and `a\tb` as `a` + `\tb` — the paren and the tab attach to the following word exactly the way a space does.
- **`\s+(?!\S)` gives up its last character.** A run of two spaces before a word splits as `" "` + `" word"`, because the lookahead forces the whitespace branch to backtrack one character so the word branch can claim it. That single rule is why byte-level BPE vocabularies are full of `Ġword` entries.

Verified by cross-checking against the pattern run through Python's `re` (with `\p{L}` and `\p{N}` translated to equivalent classes) over a 34-case corpus — every case matched, including the whole corpus as one blob with tabs, newlines, and unicode.

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

## The Tracer hook

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

`AttributionTopK` is a second gate on top of the type assertion, so a tracer that only sometimes wants attribution doesn't have to change its type to say so — returning zero turns the whole path off, recording included. The chat UI leaves it off; `-trace` and the inspect screen turn it on.

Effects are reported against the pass's own top candidates rather than the whole vocabulary. Attributing every token would mean a full LM head projection per component — sixteen extra forward passes per layer — and the question worth asking is which components pushed the tokens that were actually in contention. Folding the final norm into the unembedding directions once, instead of into every component, keeps the whole thing to a few hundred thousand multiply-adds against a 155M-parameter LM head.

The engine type-asserts for both, so a tracer that wants neither simply doesn't implement the methods and never pays for them. `trace.Tee` has to be careful here: it advertises an extension only when one of its children would actually use it, which is why there's a type per combination rather than one that implements everything.

**Why the sum is exact, and where the approximation is.** Everything after the last block is a norm and a linear map. RMSNorm's only nonlinearity is the scaling factor it computes from the whole vector — hold that at the value the *finished* stream produced and the norm becomes linear, so the output logit for a token is exactly the sum of the components' effects on it. `RMSNorm.Scale` exists to expose that factor for this. `TestAttributionSumsToTheLogit` checks the identity on both forward paths, and `TestAttributionSumsToTheLogitOnRealWeights` checks it survives the real model, where the stream reaches magnitudes in the hundreds and cancellation has room to bite.

What this measures is the *direct* path: what a component contributed by writing into the stream that reaches the LM head. A layer-2 head that matters only because layer-20 read what it wrote shows up under layer 20, not layer 2. That's the standard meaning of direct logit attribution, and it's a real limitation rather than a rounding error — reading it as "total influence" is the one way to get badly wrong conclusions from a number that always adds up.

### How the pieces are separated

```text
engine/model/  ──Tracer──▶  tools/walkthrough/     terminal walkthrough
                       └─▶  tools/trace/  ┬─ Writer  ──▶ JSONL file ─┐
                                          └─ Collector ─ in memory ──┴─▶ tools/tui (inspect screen)
```

One rule holds it together: **nothing under `engine/` imports `tools/`.** That's what the top-level folder is for. The interface lives in `engine/model/` because that's where it's consumed; every implementation lives outside. `TestEngineHasNoThirdPartyDependencies` and `TestEngineDoesNotImportTooling` parse the import graph and fail if either invariant breaks.

`Writer` and `Collector` share their event constructors, so a live view and a replayed file are looking at byte-identical data and the UI needs only one rendering path. Both copy every slice they keep, as the contract requires.

The consequences are what make this worth the indirection: the UI is testable against hand-written fixtures with no checkpoint and no inference, it can be rewritten without touching the engine, and the optimization work can restructure the hot path freely as long as the trace format holds.
