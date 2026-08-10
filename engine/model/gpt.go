// This is the Transformer stack and the full forward pass.
package model

import (
	"fmt"
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

// pass carries the state every layer needs for one forward pass: the rotary
// tables, where in the sequence this batch of tokens sits, which layer is
// running, the cache if there is one, and the tracer.
type pass struct {
	cos, sin [][]float32
	cache    *KVCache // nil for an uncached pass
	offset   int      // absolute position of x's first row
	layer    int      // -1 outside the block stack
	tr       Trace
}

func (p *pass) setLayer(i int) {
	p.layer = i
	p.tr.Layer = i
}

// layerKV returns where the current layer should store its keys and values:
// the cache when there is one, otherwise a throwaway set sized for T positions.
func (p *pass) layerKV(nKVHead, T int) *LayerKV {
	if p.cache != nil {
		return &p.cache.layers[p.layer]
	}
	return newLayerKV(nKVHead, T)
}

// Forward runs the token ids through the whole stack and returns logits
// (T, VocabSize) — one row of next-token scores per input position.
//
// This is the uncached path: it recomputes everything from position 0 and
// projects every position through the LM head. Generation doesn't use it, but
// it stays as the reference ForwardCached is checked against.
func (g *GPT) Forward(ids []int) [][]float32 {
	p := g.newPass(nil, 0, len(ids))
	x := g.stack(ids, p)

	logits := MatMul(x, g.LMHead)
	p.tr.Stage("logits", logits)
	return logits
}

// ForwardCached appends ids to the cache and returns logits for the final
// position only.
//
// Two savings over Forward. Keys and values for positions already in the cache
// aren't recomputed, so a decode step is O(T) work rather than O(T²). And the
// LM head — at 155.6M parameters, the single largest matmul in the model — runs
// on one row instead of every row.
func (g *GPT) ForwardCached(ids []int, cache *KVCache) ([]float32, error) {
	if cache == nil {
		return nil, fmt.Errorf("forward: cache is nil, use Forward for an uncached pass")
	}
	if err := cache.compatibleWith(g.Config); err != nil {
		return nil, fmt.Errorf("forward: %w", err)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("forward: no tokens to process")
	}
	for _, id := range ids {
		if id < 0 || id >= g.Config.VocabSize {
			return nil, fmt.Errorf("forward: token id %d outside vocab of %d", id, g.Config.VocabSize)
		}
	}

	offset := cache.Len()
	p := g.newPass(cache, offset, offset+len(ids))
	x := g.stack(ids, p)
	cache.n += len(ids)

	// Only the last row's logits are ever needed: it's the only position whose
	// next-token distribution hasn't already been consumed.
	logits := MatMul(x[len(x)-1:], g.LMHead)
	p.tr.Stage("logits", logits)
	return logits[0], nil
}

func (g *GPT) newPass(cache *KVCache, offset, need int) *pass {
	g.ensureRotary(need)
	return &pass{
		cos: g.cos, sin: g.sin,
		cache: cache, offset: offset, layer: -1,
		tr: Trace{Out: g.Trace, Layer: -1},
	}
}

// stack runs embeddings through every block and the final norm, stopping short
// of the LM head so callers can decide how many positions to project.
func (g *GPT) stack(ids []int, p *pass) [][]float32 {
	x := Embed(g.WTE, ids, g.Config.NEmbed)
	p.tr.Stage("token embeddings", x)

	lens := p.tr.lens()
	for i := range g.Blocks {
		p.setLayer(i)
		x = g.Blocks[i].Forward(x, p)
		if lens != nil {
			lens.LogitLens(i, g.logitsAt(x))
		}
	}

	p.setLayer(-1)
	x = g.FinalNorm.Forward(x)
	p.tr.Stage("final norm", x)
	return x
}

// logitsAt reads out the model's prediction from a mid-stack residual stream:
// apply the final norm and the LM head as if this layer were the last one.
//
// Only the final position is projected. That's the position whose next-token
// distribution we care about, and it keeps the cost to one row through the LM
// head instead of every row.
func (g *GPT) logitsAt(x [][]float32) []float32 {
	last := x[len(x)-1:]
	return MatMul(g.FinalNorm.Forward(last), g.LMHead)[0]
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
