package model

import "math"

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
