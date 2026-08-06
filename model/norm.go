package model

import "math"

// LayerNorm holds the learned per dimension scale
// and shift that GPT-2 applies after normalizing. RMSNorm had neither
type LayerNorm struct {
	Gamma []float32 //learned scale, length dim
	Beta  []float32 //learned shift, length dim
}

// NewRandomLayerNorm returns an identity initialized LayerNorm:
// gamma = 1, beta=0 means it normalizes then leaves result
func NewRandomLayerNorm(dim int) LayerNorm {
	gamma := make([]float32, dim)
	for i := range gamma {
		gamma[i] = 1
	}
	return LayerNorm{Gamma: gamma, Beta: make([]float32, dim)}
}

// Forward Normalizes each row of x (T, dim) to zero mean and unit variance
// Then scales by Gamma and shifts by Beta
func (ln *LayerNorm) Forward(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, row := range x {
		out[i] = ln.forwardVec(row)
	}
	return out
}

func (ln *LayerNorm) forwardVec(x []float32) []float32 {
	n := float64(len(x))

	var mean float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= n

	var variance float64
	for _, v := range x {
		d := float64(v) - mean
		variance += d * d
	}
	variance /= n

	// GPT-2 uses eps=1e-5, inside the sqrt alongside variance
	denom := math.Sqrt(variance + 1e-5)

	out := make([]float32, len(x))

	for i, v := range x {
		normed := (float64(v) - mean) / denom
		out[i] = float32(normed*float64(ln.Gamma[i]) + float64(ln.Beta[i]))
	}
	return out
}

// RMSNormVec normalizes a single vector by its root mean square.
// Unlike layernorm theres no mean subtraction and no learned scale / bias
// used for QK-norm (per attention head, on q and k) nd for the block level
// normalization of the residual stream
func RMSNormVec(x []float32) []float32 {
	//sum the squares , divide by lenght and add epsilon
	var sumSq float64
	for _, v := range x {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq/float64(len(x)) + 1e-6)

	out := make([]float32, len(x))

	for i, v := range x {
		out[i] = float32(float64(v) / rms)
	}
	return out
}

// RMSNorm applies RMSNormVec to every row of x (T, dim)
// Each tokens vector is normalized on its own. No mix across positions
func RMSNorm(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, row := range x {
		out[i] = RMSNormVec(row)
	}
	return out
}
