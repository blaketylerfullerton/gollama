package lenses

import (
	"math"
	"testing"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
)

// recorder collects everything the attribution path emits. It implements the
// core Tracer with no-ops, because attribution is what's under test and the
// other channels only exist here to satisfy the interface.
type recorder struct {
	topK   int
	tokens []int
	got    []recorded
	lens   []int // layers that produced a lens readout, in order
	target int
}

type recorded struct {
	layer, component int
	effects          []float32
	norm             float64
}

func (r *recorder) Stage(int, string, [][]float32)        {}
func (r *recorder) Attention(int, int, [][]float32)       {}
func (r *recorder) Rotary(int, int, []float32, []float32) {}
func (r *recorder) Note(int, string, ...any)              {}

func (r *recorder) AttributionTopK() int { return r.topK }

func (r *recorder) Attribution(layer, component int, tokens []int, effects []float32, norm float64) {
	r.tokens = append([]int(nil), tokens...)
	r.got = append(r.got, recorded{
		layer: layer, component: component,
		effects: append([]float32(nil), effects...), norm: norm,
	})
}

// lensRecorder adds the logit lens, so the ordering guarantee can be checked
// separately from attribution.
type lensRecorder struct{ recorder }

func (r *lensRecorder) LogitLens(layer int, _ []float32, target int) {
	r.lens = append(r.lens, layer)
	r.target = target
}

// The whole point of attribution is that it adds up. Every component writes into
// one residual stream, and with the final norm's scaling held at what that
// finished stream produced, the output logit for a token is exactly the sum of
// the components' effects on it.
//
// If that doesn't hold, the numbers are decorative: the per-head split through
// Wo is wrong, or a write went unrecorded, or the norm was linearized around the
// wrong vector. So this is the test that makes the feature mean anything.
func TestAttributionSumsToTheLogit(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rec := &recorder{topK: 3}
	gpt.Trace = rec

	ids := []int{1, 5, 9, 2}
	logits := gpt.Forward(ids)
	last := logits[len(logits)-1]

	if len(rec.got) == 0 {
		t.Fatal("no attribution events — the engine never asked the tracer for any")
	}

	// One write per attention head and one per MLP, in every layer, plus the
	// embedding.
	want := cfg.NLayer*(cfg.NHead+1) + 1
	if len(rec.got) != want {
		t.Errorf("recorded %d components, want %d", len(rec.got), want)
	}

	for j, id := range rec.tokens {
		var sum float64
		for _, g := range rec.got {
			sum += float64(g.effects[j])
		}
		// float32 accumulation over 8 dims x 11 components; the tolerance is
		// for the arithmetic, not for a missing term, which would be far larger.
		if diff := math.Abs(sum - float64(last[id])); diff > 1e-3 {
			t.Errorf("token %d: effects sum to %.6f, logit is %.6f (off by %.6f)",
				id, sum, last[id], diff)
		}
	}
}

// The same must hold on the cached path, which is the one generation actually
// uses and the one where only the last position is projected.
func TestAttributionSumsToTheLogitWithCache(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cache := model.NewKVCache(cfg)

	// Prefill untraced, so attribution is measured on a decode step reading real
	// history out of the cache rather than on a fresh sequence.
	if _, err := gpt.ForwardCached([]int{1, 5, 9}, cache); err != nil {
		t.Fatal(err)
	}

	rec := &recorder{topK: 2}
	gpt.Trace = rec
	last, err := gpt.ForwardCached([]int{2}, cache)
	if err != nil {
		t.Fatal(err)
	}

	for j, id := range rec.tokens {
		var sum float64
		for _, g := range rec.got {
			sum += float64(g.effects[j])
		}
		if diff := math.Abs(sum - float64(last[id])); diff > 1e-3 {
			t.Errorf("token %d: effects sum to %.6f, logit is %.6f", id, sum, last[id])
		}
	}
}

// Components are labelled, or a caller can't tell head 0 from the MLP.
func TestAttributionLabelsComponents(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{topK: 1}
	gpt.Trace = rec
	gpt.Forward([]int{1, 2})

	var heads, mlps, embeds int
	for _, g := range rec.got {
		switch {
		case g.component == model.ComponentEmbed:
			embeds++
			if g.layer != -1 {
				t.Errorf("the embedding is at layer %d, want -1", g.layer)
			}
		case g.component == model.ComponentMLP:
			mlps++
		case g.component >= 0 && g.component < cfg.NHead:
			heads++
		default:
			t.Errorf("component %d in layer %d is neither a head nor a known part",
				g.component, g.layer)
		}
		if g.norm <= 0 {
			t.Errorf("component %d in layer %d reported norm %v", g.component, g.layer, g.norm)
		}
	}
	if embeds != 1 || mlps != cfg.NLayer || heads != cfg.NLayer*cfg.NHead {
		t.Errorf("got %d embeddings, %d mlps, %d heads; want 1, %d, %d",
			embeds, mlps, heads, cfg.NLayer, cfg.NLayer*cfg.NHead)
	}
}

// A tracer that asks for zero tokens must cost nothing: no events, and no
// recording work behind the scenes either.
func TestAttributionOffWhenTopKIsZero(t *testing.T) {
	gpt, err := model.NewRandomGPT(tinyConfig())
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{topK: 0}
	gpt.Trace = rec
	gpt.Forward([]int{1, 2})

	if len(rec.got) != 0 {
		t.Errorf("recorded %d attribution events with TopK 0", len(rec.got))
	}
}

// The lens now runs after the stack, so it can report where the finally-chosen
// token ranked at each depth. Layer order still has to hold, and the target has
// to be the argmax of the real output.
func TestLogitLensRunsInLayerOrderWithTheTarget(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rec := &lensRecorder{recorder{topK: 1}}
	gpt.Trace = rec

	logits := gpt.Forward([]int{1, 2, 3})

	if len(rec.lens) != cfg.NLayer {
		t.Fatalf("got %d lens readouts, want %d", len(rec.lens), cfg.NLayer)
	}
	for i, l := range rec.lens {
		if l != i {
			t.Errorf("lens readout %d is for layer %d", i, l)
		}
	}
	if want := model.Argmax(logits[len(logits)-1]); rec.target != want {
		t.Errorf("lens target is %d, want the output argmax %d", rec.target, want)
	}
}

// Holding the scale fixed is what makes the norm linear, so a sum of writes maps
// to a sum of outputs. This checks the identity the attribution math relies on
// rather than trusting the comment above it.
func TestRMSNormIsLinearAtFixedScale(t *testing.T) {
	n := model.NewRMSNorm(4, 1e-6)
	n.Weight = []float32{0.5, 1.5, 2, 1}

	a := []float32{1, -2, 3, 0.5}
	b := []float32{-0.5, 4, 1, 2}
	sum := []float32{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}

	scale := n.Scale(sum)
	got := n.ForwardVec(sum)
	for i := range got {
		// Each part scaled by the whole's factor, then added.
		want := (float64(a[i]) + float64(b[i])) * scale * float64(n.Weight[i])
		if math.Abs(float64(got[i])-want) > 1e-6 {
			t.Errorf("dim %d: ForwardVec gave %v, the fixed-scale form gives %v", i, got[i], want)
		}
	}
}

// The tiny-model tests prove the algebra; this proves it survives the real
// thing, where the residual stream reaches magnitudes in the hundreds and the
// final norm has learned weights rather than ones. Cancellation at that scale is
// exactly where a decomposition that's only approximately right stops being
// usable, so the tolerance here is relative to the logit rather than absolute.
func TestAttributionSumsToTheLogitOnRealWeights(t *testing.T) {
	gpt := realGPT(t)
	tok := tokenizerFor(t)
	ids := tok.Encode("The capital of France is")

	rec := &recorder{topK: 3}
	gpt.Trace = rec
	defer func() { gpt.Trace = nil }()

	cache := model.NewKVCache(gpt.Config)
	last, err := gpt.ForwardCached(ids, cache)
	if err != nil {
		t.Fatal(err)
	}

	for j, id := range rec.tokens {
		var sum float64
		for _, g := range rec.got {
			sum += float64(g.effects[j])
		}
		want := float64(last[id])
		if diff := math.Abs(sum - want); diff > 1e-2*math.Abs(want) {
			t.Errorf("token %d (%q): effects sum to %.4f, logit is %.4f",
				id, tok.Decode([]int{id}), sum, want)
		}
	}
}
