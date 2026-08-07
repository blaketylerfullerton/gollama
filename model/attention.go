package model

import "math"

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
// All three are (T, HeadDim). Returns (T, HeadDim) — one output vector per
// position, each a weighted average of the value vectors at or before it.
func CausalAttention(q, k, v [][]float32) (out [][]float32, weights [][]float32) {
	T := len(q)
	headDim := len(q[0])
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	out = make([][]float32, T)
	weights = make([][]float32, T)

	for tq := 0; tq < T; tq++ {
		// Causal mask: position tq may only attend to 0..tq, never to the future.
		// Rather than building a full T x T matrix of scores and masking the
		// upper triangle with -inf, we just never compute those.
		scores := make([]float32, tq+1)
		maxScore := float32(math.Inf(-1))
		for tk := 0; tk <= tq; tk++ {
			var dot float32
			for i := 0; i < headDim; i++ {
				dot += q[tq][i] * k[tk][i]
			}

			// Scale by 1/sqrt(headDim): without it, dot products grow with
			// dimension and push softmax into a near one-hot regime.
			dot *= scale
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
		for tk := 0; tk <= tq; tk++ {
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
// x is (T, NEmbed) and expected to already be normed. cos / sin must be
// (T, HeadDim/2) — one row of rotation angles per position.
func (a *Attention) Forward(x [][]float32, cos, sin [][]float32) ([][]float32, [][][]float32) {
	T := len(x)

	q := MatMul(x, a.Wq) // (T, NHead*HeadDim)
	k := MatMul(x, a.Wk) // (T, NKVHead*HeadDim)
	v := MatMul(x, a.Wv) // (T, NKVHead*HeadDim)

	// Norm + rotate each kv head exactly once. Every query head in the group
	// reuses these — recomputing per query head would repeat the work
	// GroupSize times for no reason.
	kHeads := make([][][]float32, a.NKVHead)
	vHeads := make([][][]float32, a.NKVHead)
	for kv := 0; kv < a.NKVHead; kv++ {
		lo := kv * a.HeadDim
		kHeads[kv] = make([][]float32, T)
		vHeads[kv] = make([][]float32, T)
		for t := 0; t < T; t++ {
			// Qwen3 order is norm THEN rotary. Swapping them is self-consistent
			// with random weights but wrong against trained ones.
			kHeads[kv][t] = ApplyRotary(a.KNorm.ForwardVec(k[t][lo:lo+a.HeadDim]), cos[t], sin[t])
			vHeads[kv][t] = v[t][lo : lo+a.HeadDim] // v stays raw: no norm, no rotary
		}
	}

	out := make([][]float32, T)
	for t := range out {
		out[t] = make([]float32, a.NHead*a.HeadDim)
	}
	allWeights := make([][][]float32, a.NHead)

	groupSize := a.NHead / a.NKVHead
	for h := 0; h < a.NHead; h++ {
		lo := h * a.HeadDim
		kv := h / groupSize // which kv head this query head shares

		qh := make([][]float32, T)
		for t := 0; t < T; t++ {
			qh[t] = ApplyRotary(a.QNorm.ForwardVec(q[t][lo:lo+a.HeadDim]), cos[t], sin[t])
		}

		headOut, w := CausalAttention(qh, kHeads[kv], vHeads[kv])
		allWeights[h] = w
		for t := 0; t < T; t++ {
			copy(out[t][lo:lo+a.HeadDim], headOut[t])
		}
	}
	return MatMul(out, a.Wo), allWeights
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
