// This is the Transformer stack and the full forward pass.
package model

import (
	"math/rand/v2"
)

type Linear struct {
	Weight  []float32 // shape (out, in), row-major
	Bias    []float32 // length out; nil when the layer has no bias
	In, Out int
}

type GPT struct {
	Config    GPTConfig
	WTE       []float32 // flat (VocabSize, NEmbed) token embedding table
	Blocks    []Block
	FinalNorm RMSNorm
	LMHead    Linear

	// Trace, when set, receives every intermediate value the forward pass
	// produces. Leave it nil for normal inference.
	Trace Tracer

	// Rotary tables, grown on demand up to the longest sequence seen so far.
	cos, sin [][]float32
}

// Forward runs the token ids through the whole stack and returns logits
// (T, VocabSize) — one row of next-token scores per input position.
func (g *GPT) Forward(ids []int) [][]float32 {
	g.ensureRotary(len(ids))
	tr := Trace{Out: g.Trace, Layer: -1}

	x := Embed(g.WTE, ids, g.Config.NEmbed)
	tr.Stage("token embeddings", x)

	for i := range g.Blocks {
		tr.Layer = i
		x = g.Blocks[i].Forward(x, g.cos, g.sin, tr)
	}

	tr.Layer = -1
	x = g.FinalNorm.Forward(x)
	tr.Stage("final norm", x)

	logits := MatMul(x, g.LMHead)
	tr.Stage("logits", logits)
	return logits
}

// RotaryTables exposes the cos/sin lookup tables for inspection. They're built
// lazily, so this is only populated after at least one Forward.
func (g *GPT) RotaryTables() (cos, sin [][]float32) {
	return g.cos, g.sin
}

// ensureRotary grows the cos/sin tables if this sequence is longer than any
// we've seen. Precomputing all of SequenceLen up front would allocate tens of
// megabytes for a context we may never use.
func (g *GPT) ensureRotary(T int) {
	if T <= len(g.cos) {
		return
	}
	g.cos, g.sin = PrecomputeRotary(T, g.Config.HeadDim, g.Config.RopeBase)
}

// NewRandomGPT builds a correctly shaped model with random weights. Useful for
// exercising the forward pass before a real checkpoint is wired up — the
// shapes are real even though the numbers are noise.
func NewRandomGPT(cfg GPTConfig) (*GPT, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	blocks := make([]Block, cfg.NLayer)
	for i := range blocks {
		blocks[i] = NewRandomBlock(cfg)
	}

	wte := NewRandomEmbedding(cfg.VocabSize, cfg.NEmbed)

	// Tied embeddings: the LM head reuses the embedding table rather than
	// carrying its own (VocabSize, NEmbed) copy. Since MatMul already treats
	// Weight as (out, in) row-major, the same flat table works unchanged.
	lmHead := Linear{Weight: wte, In: cfg.NEmbed, Out: cfg.VocabSize}
	if !cfg.TieEmbed {
		lmHead = NewRandomLinear(cfg.NEmbed, cfg.VocabSize)
	}

	return &GPT{
		Config:    cfg,
		WTE:       wte,
		Blocks:    blocks,
		FinalNorm: NewRMSNorm(cfg.NEmbed, cfg.NormEps),
		LMHead:    lmHead,
	}, nil
}

func randFloats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rand.NormFloat64() * 0.02)
	}
	return out
}
