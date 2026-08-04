<div align="center">
<img src="https://i.postimg.cc/kg3gqwB1/Gemini-Generated-Image-bws52zbws52zbws5.png" alt="Go Llama" style="max-width: 600px; width: 100%;">
</div>

This repo is a rewrite by (mostly) hand of Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), but in Go.

This is purely to help me understand inference and transformers better.

Not optimal at all, just for learning. Meant to be a hackable, super simple project for understanding how LLM inference works from first principles.

Inference only — no training (yet).

## Running it

```bash
go run .
```

This walks a short prompt through every layer built so far and prints the intermediate shapes and values at each step.

## Implemented

**1. Tokenizer** — the translator sitting between human text and the numbers the model can process.

```text
"Hello, world!"  →  [Tokenizer.Encode]  →  [15496, 11, 995, 0]  →  fed into the GPT model
```

**2. Token embeddings** — each token id looks up a row in the `(vocabSize, nEmbd)` embedding table, turning a sequence of ids into a `(T, nEmbd)` matrix of vectors. Weights are randomly initialized for now.

**3. Linear layers / matmul** — `x @ W^T` projections used for Q, K, V and the output projection. Weights are stored `(out, in)` row-major to match the GPT-2 reference layout.

**4. Rotary position embeddings (RoPE)** — instead of adding a position vector, positions are encoded by *rotating* Q and K. `PrecomputeRotary` builds the cos/sin tables once; `ApplyRotary` splits each vector in half and rotates the pairs `(x[i], x[i+half])`.

**5. RMSNorm + QK-norm** — normalize by root mean square, no mean subtraction and no learned scale/bias. Applied per-head to Q and K after the rotary step.

**6. Causal attention** — single-head scaled dot-product attention with a causal mask, so position `t` can only attend to positions `≤ t`. Softmax subtracts the row max before exponentiating for numerical stability.

## Not implemented yet

- Multi-head attention (currently one head spanning the full embedding)
- MLP / feed-forward block
- Residual connections and stacked transformer blocks
- Loading real pretrained weights (everything is randomly initialized)
- Sampling loop — actually generating tokens
- KV cache

## Layout

```text
main.go          walks a prompt through every layer, printing shapes as it goes
print.go         pretty-printing helpers for vectors and matrices
tokenizer/       encode/decode, loads a tokenizer.json
model/
  embedding.go   token id → vector lookup
  linear.go      Linear layer and matmul
  rotary.go      RoPE precompute + apply
  norm.go        RMSNorm (vector and per-row)
  attention.go   causal single-head attention + softmax
  gpt.go         transformer block and forward pass (in progress)
```
