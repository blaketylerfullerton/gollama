package model

// Tracer receives intermediate values as a forward pass runs. It exists so the
// forward pass can narrate itself without main having to reach inside a block
// and reimplement the plumbing — the interesting values (attention weights,
// pre/post-rotary vectors) aren't visible from outside anyway.
//
// layer is the block index, or -1 for stages outside the block stack
// (embeddings, final norm, logits).
//
// # Contract
//
// Implementations MUST NOT retain any slice past the call that delivered it.
// The engine is free to hand over scratch buffers it intends to reuse, so a
// retained slice may be overwritten at any point afterwards. Copy anything you
// need to keep.
//
// This is deliberate. Buffer reuse is the largest allocation win available in
// the forward pass, and pushing the copy onto the tracer costs nothing when
// tracing is off — which is the normal case, since Trace is nil by default.
//
// Implementations do NOT need to be safe for concurrent use. Every call is made
// from the goroutine running the forward pass. If the matmuls are ever
// parallelized, emissions will stay on the serial path rather than fanning out.
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

// LogitLensTracer is an optional extension. A Tracer that also implements it
// receives, at every layer, the model's next-token distribution as it stands
// *at that depth* — the residual stream run through the final norm and the LM
// head early. Watching a prediction go from noise to certain across the stack is
// the clearest window into what the layers are actually doing.
//
// It's opt-in by type assertion because it isn't free: one extra LM head
// projection per layer, and the LM head is the largest matmul in the model. Only
// the final position is projected, which keeps it to roughly one extra forward
// pass worth of work overall.
type LogitLensTracer interface {
	Tracer
	LogitLens(layer int, logits []float32)
}

// Trace is the handle threaded through the forward pass. A zero Trace (nil
// Out) makes every method a no-op, so tracing costs one nil check per call
// site and nothing else — tests and benchmarks leave it unset.
type Trace struct {
	Out   Tracer
	Layer int
}

// lens returns the tracer as a LogitLensTracer, or nil if it doesn't want
// intermediate predictions.
func (t Trace) lens() LogitLensTracer {
	l, _ := t.Out.(LogitLensTracer)
	return l
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
