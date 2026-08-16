package model

import (
	"fmt"
	"math"
	"sync"
)

// Attention is one grouped-query attention layer. Under GQA the q projection
// is NHead*HeadDim wide but k and v are only NKVHead*HeadDim — several query
// heads read the same kv head, which is what shrinks the KV cache.
type Attention struct {
	Wq, Wk, Wv, Wo Linear
	QNorm, KNorm   RMSNorm // QK-norm, applied per head over HeadDim

	NHead   int
	NKVHead int
	HeadDim int
}

// CausalAttention computes single-head scaled dot product attention with a
// causal mask. q and k are expected to be already normed and rotated; v is raw.
//
// q is (Tq, HeadDim) covering absolute positions offset..offset+Tq-1, while k
// and v cover absolute positions 0..len(k)-1. During generation Tq is 1 and k
// holds the whole history, so offset is what tells a lone query where it sits
// in the sequence.
//
// Returns (Tq, HeadDim) — one output vector per query, each a weighted average
// of the value vectors at or before its position.
func CausalAttention(q, k, v [][]float32, offset int) (out [][]float32, weights [][]float32) {
	Tq := len(q)
	headDim := len(q[0])
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	if want := offset + Tq; len(k) < want {
		panic(fmt.Sprintf("CausalAttention: %d cached keys but queries reach position %d",
			len(k), want-1))
	}

	out = make([][]float32, Tq)
	weights = make([][]float32, Tq)

	for tq := 0; tq < Tq; tq++ {
		pos := offset + tq // this query's absolute position

		// Causal mask: position pos may only attend to 0..pos, never to the
		// future. Rather than building a full T x T matrix of scores and masking
		// the upper triangle with -inf, we just never compute those.
		scores := make([]float32, pos+1)
		maxScore := float32(math.Inf(-1))
		for tk := 0; tk <= pos; tk++ {
			// Scale by 1/sqrt(headDim): without it, dot products grow with
			// dimension and push softmax into a near one-hot regime.
			dot := dotProduct(q[tq], k[tk]) * scale
			scores[tk] = dot
			if dot > maxScore {
				maxScore = dot
			}
		}
		softmax(scores, maxScore)
		// returning the weights purely for visual inspection
		weights[tq] = scores

		// Weighted sum of value vectors using the attention weights.
		outVec := make([]float32, headDim)
		for tk := 0; tk <= pos; tk++ {
			w := scores[tk]
			for i := 0; i < headDim; i++ {
				outVec[i] += w * v[tk][i]
			}
		}
		out[tq] = outVec
	}
	return out, weights
}

// softmax turns scores into a probability distribution in place. Subtracting
// the max before exponentiating is the standard numerical-stability trick: it
// can't change the result mathematically.
func softmax(scores []float32, maxScore float32) {
	var sumExp float32
	for i, s := range scores {
		scores[i] = float32(math.Exp(float64(s - maxScore)))
		sumExp += scores[i]
	}
	for i := range scores {
		scores[i] /= sumExp
	}
}

// Forward runs all NHead query heads and concatenates + projects the result.
//
// x is (T, NEmbed) and expected to already be normed. It covers absolute
// positions p.offset..p.offset+T-1 — during generation that's a single row well
// past the start of the sequence.
func (a *Attention) Forward(x [][]float32, p *pass) ([][]float32, [][][]float32) {
	T := len(x)

	q := MatMul(x, a.Wq) // (T, NHead*HeadDim)
	k := MatMul(x, a.Wk) // (T, NKVHead*HeadDim)
	v := MatMul(x, a.Wv) // (T, NKVHead*HeadDim)

	// Where this layer's k/v live. With a cache they persist across calls and
	// already hold the history; without one they're built fresh and discarded.
	store := p.layerKV(a.NKVHead, T)

	// Norm + rotate each kv head's new positions exactly once. Every query head
	// in the group reuses them — recomputing per query head would repeat the
	// work GroupSize times for no reason.
	for kv := 0; kv < a.NKVHead; kv++ {
		lo := kv * a.HeadDim
		for t := 0; t < T; t++ {
			// Qwen3 order is norm THEN rotary. Swapping them is self-consistent
			// with random weights but wrong against trained ones.
			normed := a.KNorm.ForwardVec(k[t][lo : lo+a.HeadDim])
			store.append(kv, ApplyRotary(normed, p.cos[p.offset+t], p.sin[p.offset+t]),
				v[t][lo:lo+a.HeadDim]) // v stays raw: no norm, no rotary
		}
	}

	out := make([][]float32, T)
	for t := range out {
		out[t] = make([]float32, a.NHead*a.HeadDim)
	}
	allWeights := make([][][]float32, a.NHead)

	groupSize := a.NHead / a.NKVHead
	tracing := p.tr.On()

	// Each query head is independent of the others (they only read q/store,
	// never write shared state beyond their own slice of out/allWeights), so
	// they run concurrently. Tracing forces the sequential path: p.tr.Rotary
	// isn't safe to call from multiple goroutines at once.
	if !tracing {
		var wg sync.WaitGroup
		wg.Add(a.NHead)
		for h := 0; h < a.NHead; h++ {
			go func(h int) {
				defer wg.Done()
				attentionHead(a, h, groupSize, q, store, p, T, out, allWeights, false)
			}(h)
		}
		wg.Wait()
	} else {
		for h := 0; h < a.NHead; h++ {
			attentionHead(a, h, groupSize, q, store, p, T, out, allWeights, true)
		}
	}

	// Each head's own write into the residual stream, recorded before the heads
	// are summed and become indistinguishable. Wo mixes across the concatenated
	// heads, but it's linear, so the head's share is its slice of the input
	// against the matching columns — see headContribution.
	if p.wantWrites {
		for h := 0; h < a.NHead; h++ {
			p.record(h, a.headContribution(h, out[T-1]))
		}
	}
	return MatMul(out, a.Wo), allWeights
}

// headContribution projects one head's slice of the concatenated attention
// output through the matching columns of Wo, giving that head's additive share
// of the block's attention output.
//
// concat is a single position's full NHead*HeadDim row. Wo's bias, if a model
// ever has one, belongs to no head and is left out of every share.
func (a *Attention) headContribution(h int, concat []float32) []float32 {
	lo := h * a.HeadDim
	c := make([]float32, a.Wo.Out)
	for o := range c {
		row := a.Wo.Weight[o*a.Wo.In : (o+1)*a.Wo.In]
		c[o] = dotProduct(concat[lo:lo+a.HeadDim], row[lo:lo+a.HeadDim])
	}
	return c
}

// attentionHead computes one query head's attention output and writes it into
// its slice of out (and allWeights). Safe to call concurrently across
// different h values as long as trace is false.
func attentionHead(a *Attention, h, groupSize int, q [][]float32, store *LayerKV, p *pass, T int, out [][]float32, allWeights [][][]float32, trace bool) {
	lo := h * a.HeadDim
	kv := h / groupSize // which kv head this query head shares

	qh := make([][]float32, T)
	for t := 0; t < T; t++ {
		pos := p.offset + t
		normed := a.QNorm.ForwardVec(q[t][lo : lo+a.HeadDim])
		qh[t] = ApplyRotary(normed, p.cos[pos], p.sin[pos])
		// Report the last position: it has rotated the furthest, so the
		// change is most visible there.
		if trace && t == T-1 {
			p.tr.Rotary(h, normed, qh[t])
		}
	}

	headOut, w := CausalAttention(qh, store.K[kv], store.V[kv], p.offset)
	allWeights[h] = w
	for t := 0; t < T; t++ {
		copy(out[t][lo:lo+a.HeadDim], headOut[t])
	}
}

func NewRandomAttention(cfg GPTConfig) Attention {
	return Attention{
		Wq:      NewRandomLinear(cfg.NEmbed, cfg.QOut()),
		Wk:      NewRandomLinear(cfg.NEmbed, cfg.KVOut()),
		Wv:      NewRandomLinear(cfg.NEmbed, cfg.KVOut()),
		Wo:      NewRandomLinear(cfg.QOut(), cfg.NEmbed),
		QNorm:   NewRMSNorm(cfg.HeadDim, cfg.NormEps),
		KNorm:   NewRMSNorm(cfg.HeadDim, cfg.NormEps),
		NHead:   cfg.NHead,
		NKVHead: cfg.NKVHead,
		HeadDim: cfg.HeadDim,
	}
}
