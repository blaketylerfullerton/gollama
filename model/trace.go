package model

// Tracer receives intermediate values as a forward pass runs. It exists so the
// forward pass can narrate itself without main having to reach inside a block
// and reimplement the plumbing — the interesting values (attention weights,
// pre/post-rotary vectors) aren't visible from outside anyway.
//
// layer is the block index, or -1 for stages outside the block stack
// (embeddings, final norm, logits).
type Tracer interface {
	// Stage reports the residual stream, or any (T, dim) matrix, after a step.
	Stage(layer int, name string, x [][]float32)
	// Attention reports one head's causal attention weights. Row tq has
	// tq+1 entries — everything above the diagonal is masked, not zeroed.
	Attention(layer, head int, weights [][]float32)
	// Rotary reports one head's q vector either side of the rotation, so a
	// reader can see that the length is preserved.
	Rotary(layer, head int, before, after []float32)
	// Note reports a piece of commentary that isn't a tensor.
	Note(layer int, format string, args ...any)
}

// Trace is the handle threaded through the forward pass. A zero Trace (nil
// Out) makes every method a no-op, so tracing costs one nil check per call
// site and nothing else — tests and benchmarks leave it unset.
type Trace struct {
	Out   Tracer
	Layer int
}

// On reports whether anyone is listening. Guard expensive trace-only work with
// this rather than computing it and throwing it away.
func (t Trace) On() bool { return t.Out != nil }

func (t Trace) Stage(name string, x [][]float32) {
	if t.Out != nil {
		t.Out.Stage(t.Layer, name, x)
	}
}

func (t Trace) Attention(head int, weights [][]float32) {
	if t.Out != nil {
		t.Out.Attention(t.Layer, head, weights)
	}
}

func (t Trace) Rotary(head int, before, after []float32) {
	if t.Out != nil {
		t.Out.Rotary(t.Layer, head, before, after)
	}
}

func (t Trace) Note(format string, args ...any) {
	if t.Out != nil {
		t.Out.Note(t.Layer, format, args...)
	}
}
