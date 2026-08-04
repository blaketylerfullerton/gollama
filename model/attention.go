package model

import "math"

// CausalAttention computes single-head scaled dot product attention with a causal mask
// q and k are expected to be already rotated and QK - normed;
// v is raw (no rotary, no norm)

// All three are (T, headDim). Returns (T, headDim) - one output vector per position
// Each weighted average of the value vectors at or before it
func CausalAttention(q, k, v [][]float32) (out [][]float32, weights [][]float32) {
	T := len(q)
	headDim := len(q[0])
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	out = make([][]float32, T)
	weights = make([][]float32, T)

	for tq := 0; tq < T; tq++ {
		// cuasal mask: position tq may only attent to 0..tq, never to the future
		// rather than buidling a full T x T matrix of scores and
		// masking the upper triangle with -inf, we just never compute those
		scores := make([]float32, tq+1)
		maxScore := float32(math.Inf(-1))
		for tk := 0; tk <= tq; tk++ {
			var dot float32
			for i := 0; i < headDim; i++ {
				dot += q[tq][i] * k[tk][i]
			}

			// Scale by 1/sqrt(headDim): without it, dot products grow with
			// dimension and push softmax into a near one hot regime.
			dot *= scale
			scores[tk] = dot
			if dot > maxScore {
				maxScore = dot
			}
		}
		softmax(scores, maxScore)
		//returning the weights purely for visual
		weights[tq] = scores

		//Weighted sum of value vectors using the attention weights.
		outVec := make([]float32, headDim)
		for tk := 0; tk <= tq; tk++ {
			w := scores[tk]
			for i := 0; i < headDim; i++ {
				outVec[i] += w * v[tk][i]
			}
			out[tq] = outVec
		}
	}
	return out, weights
}

// softmax turns scores into a probabliituy distribution in place.
// subtracting the max before exponentiating is the standard numerican stability truck: it cant change the result mathemitcallyu
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
