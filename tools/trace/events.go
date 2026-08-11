package trace

import (
	"fmt"
	"math"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
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

func lensEvent(opts Opts, layer int, logits []float32, target int) Event {
	cands := model.TopCandidates(logits, 1.0, opts.TopK)
	top := make([]Candidate, len(cands))
	for i, c := range cands {
		top[i] = Candidate{ID: c.ID, Text: opts.text(c.ID), Prob: c.Prob}
	}

	e := Event{Kind: KindLogitLens, Layer: layer, Top: top, Entropy: entropy(logits)}
	if target >= 0 && target < len(logits) {
		// Where the answer stood at this depth. Top-k can't say: a token at
		// rank 40 and a token that never appears look identical from the top of
		// the list, and the difference between them is the whole story of how a
		// prediction forms.
		e.TargetID = target
		e.TargetText = opts.text(target)
		e.TargetRank = rank(logits, target)
		e.TargetProb = probOf(logits, target)
	}
	return e
}

func attributionEvent(opts Opts, layer, component int, tokens []int, effects []float32, norm float64) Event {
	e := Event{Kind: KindAttribution, Layer: layer, Norm: norm}
	switch component {
	case model.ComponentMLP:
		e.Component = ComponentMLP
	case model.ComponentEmbed:
		e.Component = ComponentEmbed
	default:
		e.Component, e.Head = ComponentHead, component
	}

	e.Effects = make([]Effect, len(tokens))
	for i, id := range tokens {
		e.Effects[i] = Effect{ID: id, Text: opts.text(id), Logit: effects[i]}
	}
	return e
}

// text resolves a token id, or returns "" when the trace was made without a
// vocabulary.
func (o Opts) text(id int) string {
	if o.Vocab == nil {
		return ""
	}
	return o.Vocab(id)
}

// entropy is the Shannon entropy of the softmax over the whole vocabulary, in
// nats. It's the one number that says how committed the model is at this depth
// without reference to any particular token: near log(vocab) is a model with no
// idea, near zero is one that has made up its mind.
func entropy(logits []float32) float64 {
	maxL, sum := maxLogit(logits), 0.0
	for _, l := range logits {
		sum += math.Exp(float64(l) - maxL)
	}
	logZ := maxL + math.Log(sum)

	var h float64
	for _, l := range logits {
		logp := float64(l) - logZ
		h -= math.Exp(logp) * logp
	}
	return h
}

// rank is the 1-based position of id when logits are sorted descending. Ties
// resolve in id's favour, which matches how a sort would place the first of
// several equal values.
func rank(logits []float32, id int) int {
	r := 1
	for i, l := range logits {
		if l > logits[id] || (l == logits[id] && i < id) {
			r++
		}
	}
	return r
}

func probOf(logits []float32, id int) float64 {
	maxL, sum := maxLogit(logits), 0.0
	for _, l := range logits {
		sum += math.Exp(float64(l) - maxL)
	}
	return math.Exp(float64(logits[id])-maxL) / sum
}

func maxLogit(logits []float32) float64 {
	m := float64(logits[0])
	for _, l := range logits[1:] {
		if float64(l) > m {
			m = float64(l)
		}
	}
	return m
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
