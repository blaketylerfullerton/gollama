# Layout

The full annotated file tree. See the [README](../README.md#layout) for the two-folder philosophy behind it.

```text
main.go              flag parsing, the printed walkthrough (-no-splash / piped),
                     and setup() — loads a checkpoint, or falls back to a
                     random model when one isn't there
chat.go              the interactive path: wires the picker's choice into a
                     live *model.GPT and turns typed lines into streamed tokens
                     plus the per-token attention/candidate summaries the
                     chat screen's inspect tab shows
inspect_engine.go    the same wiring for the inspect screen: loads a
                     checkpoint, runs traced prefill/decode passes (optionally
                     with a head ablated), and answers the live tokenization
                     preview — used to be its own binary, cmd/inspect
watermark_demo.go    the -watermark flag's printed comparison: plain vs.
                     tournament-sampled generation, scored by tools/watermark
watermark_engine.go  the same comparison wired into the Watermark screen

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
  tui/               the whole screen flow (bubbletea, lipgloss)
    root.go          Root — one bubbletea model that owns every screen and
                     decides which is visible, so a screen can be left rather
                     than only ever ended
    welcome.go       screen 1: this machine and what's on disk to run on it,
                     and the tool menu — Ablation, Attention, Attribution,
                     Logit Lens, Watermark, Chat
    picker.go        screen 2: every known model, memory cost against this
                     machine, and a fits/recommended/too-large verdict —
                     shared by every tool picked on the welcome menu
    catalog.go       the model list itself — names, architectures, what's
                     installed under a checkpoint root
    download.go      screen: fetches a picked model's weights from HuggingFace
                     (via tools/hf) and shows live progress, in place of
                     sending you to another terminal to run huggingface-cli
    chat.go          screen: the conversation, plus the inspect tab
    inspect.go       screen: the pass inspector — logit lens, attention,
                     attribution, ablation, stages — ported in from what used
                     to be the separate cmd/inspect binary
    inspect_views.go the five views' rendering
    about.go         the "what is GoLlama" page
    history.go       screen: "past conversations" — read-only playback of
                     whatever tools/history has saved
    layout.go        the frame every screen shares: header, toolbar, panels
    llama.go         the animated ASCII llama
    wordmark.go      the "GoLlama" title art
    style.go         shared lipgloss styles built on tools/amber
  amber/             the palette: one brightness ramp for data (attention
                     weights, probabilities, memory gauges) kept separate from
                     the neutral ramp used for chrome — see the package doc
  hf/                fetches a checkpoint's config/tokenizer/weights straight
                     from huggingface.co, resuming a partial multi-shard
                     download instead of restarting it
  history/           persists chat transcripts as JSON, one file per
                     conversation, under ~/.gollama/history
  sysinfo/           what hardware this is about to run on
  watermark/         SynthID-Text-style tournament sampling and its detector —
                     see docs/watermarking.md

docs/                deep dives out of the README's way — engine internals,
                     the inspect screen's four views, watermarking, testing,
                     and this file

.github/workflows/   CI: build, vet, gofmt -l, golangci-lint, and go test ./... on
                     every push to main
.golangci.yml        lint config for CI's golangci-lint step — the standard
                     linter set, nothing project-specific yet
```
