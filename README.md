<div align="center">
<img src="https://i.postimg.cc/kg3gqwB1/Gemini-Generated-Image-bws52zbws52zbws5.png" alt="Go Llama" style="max-width: 600px; width: 100%;">
</div>

This repo started as a rewrite by (mostly) hand of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), but in Go. It has since moved to the **Qwen3** architecture so that real pretrained checkpoints will load.

This is purely to help me understand inference and transformers better.

Not optimal at all, just for learning. Meant to be a hackable, super simple project for understanding how LLM inference works from first principles.

Inference only — no training (yet). Standard library only, no dependencies.

## Running it

```bash
go run .
```

The forward pass narrates itself. `main.go` walks a prompt through every stage and prints the shapes, intermediate vectors, and attention weights as it goes.

The best bit is the attention grid, which makes causal masking obvious:

```text
attention weights — layer 0, head 0 (each row attends across the columns)
               Hello        ,   _world        . _Testing   _token  ization   _layer
  Hello        1.000        ·        ·        ·        ·        ·        ·        ·
  ,            0.630    0.370        ·        ·        ·        ·        ·        ·
  _world       0.401    0.314    0.286        ·        ·        ·        ·        ·
  .            0.098    0.096    0.126    0.680        ·        ·        ·        ·
  _Testing     0.032    0.067    0.682    0.033    0.186        ·        ·        ·
  _token       0.144    0.441    0.044    0.185    0.141    0.045        ·        ·
  ization      0.112    0.624    0.112    0.017    0.014    0.020    0.100        ·
  _layer       0.299    0.311    0.081    0.031    0.035    0.055    0.075    0.113
  · = masked. Each row sums to 1, and token 0 can only ever attend to itself
```

`main.go` runs with random weights, so your numbers will differ and the predictions are meaningless — the mechanics are the point.

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

Set `gpt.Trace` and you get the walkthrough. Leave it nil — which is what tests and any real inference do — and every call is a no-op behind a single nil check. That way the pedagogy lives in `print.go` instead of being tangled through the forward pass, and things you can't see from outside a block (attention weights, pre/post-rotary vectors) are still reachable.

## Architecture

Qwen3, which is very close to Llama:

**Grouped-query attention (GQA)** — several query heads share one key/value head, so the KV cache is `NHead/NKVHead` times smaller. `k` and `v` project to `NKVHead*HeadDim` while `q` projects to `NHead*HeadDim`.

**Rotary position embeddings (RoPE)** — positions are encoded by *rotating* q and k rather than adding a position vector. `PrecomputeRotary` builds the cos/sin tables once; `ApplyRotary` splits each vector in half and rotates the pairs `(x[i], x[i+half])`. Signs match HuggingFace's `rotate_half`. `rope_theta` comes from config (Qwen3 uses 1e6, Llama 2 uses 10000).

**RMSNorm** — normalize by root mean square, no mean subtraction, no bias, but *with* a learned per-dimension scale. Used before attention, before the MLP, and once before the LM head.

**QK-norm** — an RMSNorm applied per attention head to q and k, over `HeadDim`. It runs **before** the rotary step, matching Qwen3. Llama doesn't have this at all.

**SwiGLU MLP** — three matrices rather than two: `down(silu(gate(x)) * up(x))`. This is why the intermediate width isn't the classic `4*NEmbed`.

**Causal attention** — scaled dot-product with a causal mask, so position `t` only attends to positions `≤ t`. Rather than building a full `T x T` score matrix and masking the upper triangle with `-inf`, the masked scores are simply never computed. Softmax subtracts the row max before exponentiating for numerical stability.

**Tied embeddings** — small Qwen3 models omit `lm_head.weight` and reuse the embedding table. Detected from the checkpoint rather than trusting the config flag.

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

No transposes needed anywhere: PyTorch `nn.Linear` already stores weights `(out, in)` row-major, which is exactly what `Linear.Weight` wants. (GPT-2's `Conv1D` stores them transposed, which is one of several reasons this targets Qwen3 instead.)

To get a checkpoint:

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./checkpoints/qwen3-0.6b
```

## Tests

```bash
go test ./...
```

The safetensors and loader tests build synthetic checkpoints on disk using real HuggingFace tensor names, over a tiny config that deliberately keeps `HeadDim != NEmbed/NHead` and `NKVHead < NHead` — so anything that wrongly derives head dim, or confuses query width with kv width, fails there rather than in a 600MB checkpoint.

`TestForwardIsCausal` is the one I'd point at: truncating the input must not change the logits for the positions that remain, because no position may attend to the future. It's the strongest correctness property available without a reference implementation to compare against.

## Not implemented yet

- **Generation loop** — sampling a token, appending it, and re-running. Right now `Forward` gives you one pass and `main.go` prints the resulting distribution.
- **KV cache** — every step currently recomputes the whole prefix, which is `O(T²)` work per token.
- **A Qwen3-compatible tokenizer.** The current one is GPT-2 shaped. The pretokenizer regex in `tokenizer.json` uses negative lookahead (`\s+(?!\S)`), which Go's stdlib `regexp` (RE2) cannot compile by design, so this needs a hand-written splitter.
- **Performance.** Nothing is optimized. `[][]float32` is a pointer chase per row, matmul is a naive triple loop, and weights are widened to float32 on load.
- **Verification against a reference.** No golden-logits test yet, so the rotary sign convention and QK-norm ordering are reasoned from the HuggingFace source rather than proven.

Despite the name, it doesn't run Llama yet — but it's close. Llama needs optional QK-norm (it has none) and RoPE scaling for 3.1+. Everything else is already config-driven.

## Layout

```text
main.go              walks a prompt through every stage, printing as it goes
print.go             the walkthrough Tracer, plus vector/matrix pretty-printers
tokenizer/           byte-level BPE encode/decode, loads a tokenizer.json
model/
  config.go          GPTConfig + HuggingFace config.json loader + validation
  safetensors.go     .safetensors reader, F32/BF16/F16, sharded checkpoints
  loader.go          maps Qwen3 tensor names onto the structs
  trace.go           the Tracer hook the forward pass narrates through
  gpt.go             the model, the full forward pass, rotary table management
  block.go           one transformer layer: pre-norm attention + pre-norm MLP
  attention.go       grouped-query attention, causal attention, softmax
  mlp.go             SwiGLU feed-forward
  norm.go            RMSNorm with a learned scale
  rotary.go          RoPE precompute + apply
  linear.go          Linear layer and matmul
  embedding.go       token id → vector lookup
  ops.go             elementwise helpers (residual add)
```
