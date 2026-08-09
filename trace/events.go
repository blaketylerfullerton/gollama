package trace

import (
	"fmt"
	"math"

	"github.com/blaketylerfullerton/GoLlama/model"
)

// Event construction lives here so the file Writer and the in-memory Collector
// build identical events. Every constructor copies whatever it keeps: the Tracer
// contract lets the engine reuse the buffers it hands over, so retaining one
// would mean reading memory that has since been overwritten.

// stageEvent reports false when there's nothing worth recording.
func stageEvent(opts Opts, layer int, name string, x [][]float32) (Event, bool) {
	if len(x) == 0 {
		return Event{}, false
	}
	// The last row is the interesting one: it's the position whose prediction
	// everything downstream is about.
	return Event{
		Kind: KindStage, Layer: layer, Name: name,
		Tokens: len(x), Dims: len(x[0]),
		MeanNorm: meanNorm(x),
		Preview:  clip(x[len(x)-1], opts.PreviewDims),
	}, true
}

// attentionEvent reports false for sequences past the recording limit. Weights
// are O(T²) per head per layer, so a long prompt would otherwise dominate.
func attentionEvent(opts Opts, layer, head int, weights [][]float32) (Event, bool) {
	if len(weights) > opts.MaxAttentionTokens {
		return Event{}, false
	}
	rows := make([][]float32, len(weights))
	for i, r := range weights {
		rows[i] = append([]float32(nil), r...)
	}
	return Event{Kind: KindAttention, Layer: layer, Head: head, Weights: rows}, true
}

func rotaryEvent(opts Opts, layer, head int, before, after []float32) Event {
	return Event{
		Kind: KindRotary, Layer: layer, Head: head,
		Before: clip(before, opts.PreviewDims),
		After:  clip(after, opts.PreviewDims),
		// Rotation preserves length; recording both norms is what lets a reader
		// verify that rather than take it on trust.
		NormIn: norm(before), NormOut: norm(after), CosSim: cosSim(before, after),
	}
}

func noteEvent(layer int, format string, args ...any) Event {
	return Event{Kind: KindNote, Layer: layer, Text: fmt.Sprintf(format, args...)}
}

func lensEvent(opts Opts, layer int, logits []float32) Event {
	cands := model.TopCandidates(logits, 1.0, opts.TopK)
	top := make([]Candidate, len(cands))
	for i, c := range cands {
		text := ""
		if opts.Vocab != nil {
			text = opts.Vocab(c.ID)
		}
		top[i] = Candidate{ID: c.ID, Text: text, Prob: c.Prob}
	}
	return Event{Kind: KindLogitLens, Layer: layer, Top: top}
}

// clip copies at most n leading values, so the engine stays free to reuse the
// buffer it just handed us.
func clip(v []float32, n int) []float32 {
	if n > len(v) {
		n = len(v)
	}
	return append([]float32(nil), v[:n]...)
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func meanNorm(x [][]float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var sum float64
	for _, row := range x {
		sum += norm(row)
	}
	return sum / float64(len(x))
}

func cosSim(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	if d := norm(a) * norm(b); d != 0 {
		return dot / d
	}
	return 0
}
