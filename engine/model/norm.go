package model

import "math"

// RMSNorm normalizes a vector by its root mean square, then scales by a
// learned per-dimension weight. Unlike LayerNorm there is no mean subtraction
// and no bias — Qwen3 (and Llama) use this everywhere: before attention,
// before the MLP, on q and k inside each head, and once more before the LM head.
type RMSNorm struct {
	Weight []float32 // learned scale, length dim
	Eps    float64
}

// NewRMSNorm returns an identity-initialized norm: weight = 1 normalizes and
// leaves the result alone, which is what you want before real weights load.
func NewRMSNorm(dim int, eps float64) RMSNorm {
	w := make([]float32, dim)
	for i := range w {
		w[i] = 1
	}
	return RMSNorm{Weight: w, Eps: eps}
}

// Scale is the factor ForwardVec divides x by, ahead of the learned weight:
// 1/rms(x). It depends on the whole vector, which is the only nonlinear thing
// this norm does.
//
// Attribution needs it separately because holding it fixed at what the finished
// residual stream produced makes the norm linear in its input, and a sum of
// contributions then maps to a sum of logits. Everything else calls ForwardVec.
func (n *RMSNorm) Scale(x []float32) float64 {
	var sumSq float64
	for _, v := range x {
		sumSq += float64(v) * float64(v)
	}
	return 1 / math.Sqrt(sumSq/float64(len(x))+n.Eps)
}

// ForwardVec normalizes a single vector. Used directly for QK-norm, where the
// norm applies per attention head over HeadDim rather than over the whole row.
func (n *RMSNorm) ForwardVec(x []float32) []float32 {
	scale := n.Scale(x)

	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v) * scale * float64(n.Weight[i]))
	}
	return out
}

// Forward applies ForwardVec to every row of x (T, dim). Each token's vector
// is normalized on its own — nothing mixes across positions.
func (n *RMSNorm) Forward(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, row := range x {
		out[i] = n.ForwardVec(row)
	}
	return out
}
