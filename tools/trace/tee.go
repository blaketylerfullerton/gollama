package trace

import (
	"github.com/blaketylerfullerton/GoLlama/engine/model"
)

// Tee fans trace events out to several tracers, so one run can both print
// a walkthrough and write a trace file.
//
// The subtlety is the logit lens. The engine decides whether to compute it by
// type-asserting the tracer to model.LogitLensTracer, and it costs one extra LM
// head projection per layer — the largest matmul in the model. So the returned
// value implements that interface only when at least one child would actually
// use it. A single child is returned unwrapped, which preserves its exact
// interface set.
func Tee(children ...model.Tracer) model.Tracer {
	switch len(children) {
	case 0:
		return nil
	case 1:
		return children[0]
	}
	t := tee(children)
	for _, c := range children {
		if _, ok := c.(model.LogitLensTracer); ok {
			return t // tee implements LogitLensTracer
		}
	}
	return quiet{t} // no child wants the lens, so don't advertise it
}

// tee forwards to every child, including the logit lens.
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

func (t tee) LogitLens(layer int, logits []float32) {
	for _, out := range t {
		if lens, ok := out.(model.LogitLensTracer); ok {
			lens.LogitLens(layer, logits)
		}
	}
}

// quiet forwards the four core methods and nothing else. The methods are
// written out rather than promoted from an embedded tee, because embedding would
// also promote LogitLens and defeat the point of this type.
type quiet struct{ children tee }

func (q quiet) Stage(layer int, name string, x [][]float32) {
	q.children.Stage(layer, name, x)
}

func (q quiet) Attention(layer, head int, weights [][]float32) {
	q.children.Attention(layer, head, weights)
}

func (q quiet) Rotary(layer, head int, before, after []float32) {
	q.children.Rotary(layer, head, before, after)
}

func (q quiet) Note(layer int, format string, args ...any) {
	q.children.Note(layer, format, args...)
}
