package trace

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
)

// Opts tunes what a Writer or Collector records.
type Opts struct {
	// PreviewDims is how many leading dimensions of a vector to store. Whole
	// rows would make traces enormous for no benefit — 1024 floats per stage
	// per layer, and nobody reads past the first handful.
	PreviewDims int
	// MaxAttentionTokens skips attention weights once a sequence is longer than
	// this. The weights are O(T²) per head per layer, so a long prompt would
	// otherwise dominate.
	MaxAttentionTokens int
	// TopK is how many candidates to keep per logit-lens readout.
	TopK int
	// Vocab, when set, resolves token ids to text so a reader needs no
	// tokenizer. Returning "" for unknown ids is fine.
	Vocab func(id int) string
}

func (o Opts) withDefaults() Opts {
	if o.PreviewDims <= 0 {
		o.PreviewDims = 16
	}
	if o.MaxAttentionTokens <= 0 {
		o.MaxAttentionTokens = 32
	}
	if o.TopK <= 0 {
		o.TopK = 5
	}
	return o
}

// Writer implements model.Tracer (and model.LogitLensTracer) by appending JSON
// Lines to an io.Writer.
type Writer struct {
	enc  *json.Encoder
	buf  *bufio.Writer
	opts Opts
	seq  int
	err  error
}

// NewWriter writes the header and returns a Writer ready to receive events.
func NewWriter(w io.Writer, h Header, opts Opts) (*Writer, error) {
	h.Version = FormatVersion
	buf := bufio.NewWriter(w)
	enc := json.NewEncoder(buf)
	if err := enc.Encode(h); err != nil {
		return nil, fmt.Errorf("writing trace header: %w", err)
	}
	return &Writer{enc: enc, buf: buf, opts: opts.withDefaults()}, nil
}

// Close flushes buffered output and reports the first error encountered,
// whether from a write during tracing or from the flush itself.
func (w *Writer) Close() error {
	if err := w.buf.Flush(); err != nil && w.err == nil {
		w.err = err
	}
	return w.err
}

// Events is how many events were recorded.
func (w *Writer) Events() int { return w.seq }

// emit records one event. Write errors are latched rather than returned,
// because a tracer sits on the forward pass and must not be able to fail it.
func (w *Writer) emit(e Event) {
	if w.err != nil {
		return
	}
	e.Seq = w.seq
	w.seq++
	if err := w.enc.Encode(e); err != nil && w.err == nil {
		w.err = err
	}
}

func (w *Writer) Stage(layer int, name string, x [][]float32) {
	if e, ok := stageEvent(w.opts, layer, name, x); ok {
		w.emit(e)
	}
}

func (w *Writer) Attention(layer, head int, weights [][]float32) {
	if e, ok := attentionEvent(w.opts, layer, head, weights); ok {
		w.emit(e)
	}
}

func (w *Writer) Rotary(layer, head int, before, after []float32) {
	w.emit(rotaryEvent(w.opts, layer, head, before, after))
}

func (w *Writer) Note(layer int, format string, args ...any) {
	w.emit(noteEvent(layer, format, args...))
}

// LogitLens records what the model would predict if this layer were the last.
func (w *Writer) LogitLens(layer int, logits []float32) {
	w.emit(lensEvent(w.opts, layer, logits))
}
