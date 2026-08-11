package trace

import (
	"github.com/blaketylerfullerton/GoLlama/engine/model"
)

// Tee fans trace events out to several tracers, so one run can both print
// a walkthrough and write a trace file.
//
// The subtlety is the optional extensions. The engine decides whether to compute
// the logit lens and the attribution by type-asserting the tracer, and both cost
// real work — an LM head projection per layer, and a Wo-sized matmul per layer.
// So the returned value must advertise an extension only when some child would
// actually use it, which is why there's a type per combination rather than one
// that implements everything. A single child is returned unwrapped, which
// preserves its exact interface set.
func Tee(children ...model.Tracer) model.Tracer {
	switch len(children) {
	case 0:
		return nil
	case 1:
		return children[0]
	}

	t := tee(children)
	var wantLens, wantAttrib bool
	for _, c := range children {
		if _, ok := c.(model.LogitLensTracer); ok {
			wantLens = true
		}
		if a, ok := c.(model.AttributionTracer); ok && a.AttributionTopK() > 0 {
			wantAttrib = true
		}
	}

	switch {
	case wantLens && wantAttrib:
		return teeBoth{t}
	case wantLens:
		return teeLens{t}
	case wantAttrib:
		return teeAttrib{t}
	default:
		return t
	}
}

// tee forwards the four core methods and nothing else.
type tee []model.Tracer

func (t tee) Stage(layer int, name string, x [][]float32) {
	for _, out := range t {
		out.Stage(layer, name, x)
	}
}

func (t tee) Attention(layer, head int, weights [][]float32) {
	for _, out := range t {
		out.Attention(layer, head, weights)
	}
}

func (t tee) Rotary(layer, head int, before, after []float32) {
	for _, out := range t {
		out.Rotary(layer, head, before, after)
	}
}

func (t tee) Note(layer int, format string, args ...any) {
	for _, out := range t {
		out.Note(layer, format, args...)
	}
}

// logitLens and attribution hold the forwarding logic, so the wrapper types
// below differ only in which of them they expose.
func (t tee) logitLens(layer int, logits []float32, target int) {
	for _, out := range t {
		if lens, ok := out.(model.LogitLensTracer); ok {
			lens.LogitLens(layer, logits, target)
		}
	}
}

// attributionTopK is the largest any child asked for. A child that wanted fewer
// receives more than it asked for, which is harmless — the alternative is
// running attribution once per distinct k.
func (t tee) attributionTopK() int {
	k := 0
	for _, out := range t {
		if a, ok := out.(model.AttributionTracer); ok && a.AttributionTopK() > k {
			k = a.AttributionTopK()
		}
	}
	return k
}

func (t tee) attribution(layer, component int, tokens []int, effects []float32, norm float64) {
	for _, out := range t {
		if a, ok := out.(model.AttributionTracer); ok && a.AttributionTopK() > 0 {
			a.Attribution(layer, component, tokens, effects, norm)
		}
	}
}

// The wrappers embed tee, which promotes the four core methods, and lift only
// the extensions their children asked for. Embedding is what keeps this to one
// line per method instead of four types' worth of forwarding.
type teeLens struct{ tee }

func (t teeLens) LogitLens(layer int, logits []float32, target int) {
	t.logitLens(layer, logits, target)
}

type teeAttrib struct{ tee }

func (t teeAttrib) AttributionTopK() int { return t.attributionTopK() }
func (t teeAttrib) Attribution(layer, component int, tokens []int, effects []float32, norm float64) {
	t.attribution(layer, component, tokens, effects, norm)
}

type teeBoth struct{ tee }

func (t teeBoth) LogitLens(layer int, logits []float32, target int) {
	t.logitLens(layer, logits, target)
}
func (t teeBoth) AttributionTopK() int { return t.attributionTopK() }
func (t teeBoth) Attribution(layer, component int, tokens []int, effects []float32, norm float64) {
	t.attribution(layer, component, tokens, effects, norm)
}
