package model

import "math"

// precomputeRotary builds cos / sin lookup tables for every position
// headDim is the size of one attensinots head vector (not the full embedding)
// Returns two (seqLen, headDim/2) tables - cos[t]/sin[t] give the rotation angles for position t
func PrecomputeRotary(seqLen, headDim int, base float64) (cos, sin [][]float32) {
	halfDim := headDim / 2
	invFreq := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		channel := float64(2 * i)
		invFreq[i] = 1.0 / math.Pow(base, channel/float64(headDim))
	}

	cos = make([][]float32, seqLen)
	sin = make([][]float32, seqLen)

	for t := 0; t < seqLen; t++ {
		cos[t] = make([]float32, halfDim)
		sin[t] = make([]float32, halfDim)
		for i := 0; i < halfDim; i++ {
			freq := float64(t) * invFreq[i]
			cos[t][i] = float32(math.Cos(freq))
			sin[t][i] = float32(math.Sin(freq))
		}
	}
	return cos, sin
}

// applyRotary rotates a single vector (e.g one atentions heads q or k)
// using the cos / sin for its position. Splits the cevtor in half and rotates each pair (x[i], x[i+half] together)
func ApplyRotary(x []float32, cos, sin []float32) []float32 {
	d := len(x) / 2
	out := make([]float32, len(x))
	for i := 0; i < d; i++ {
		x1, x2 := x[i], x[i+d]
		out[i] = x1*cos[i] + x2*sin[i]
		out[i+d] = -x1*sin[i] + x2*cos[i]
	}
	return out
}
