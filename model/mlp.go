package model

import "math"

// MLP is the feed forward half of a transformer block. Expands to 4x the
// Embedding dim. apply nonlinerarity, project back down
func MLP(x [][]float32, wfc, wproj Linear) [][]float32 {
	h := MatMul(x, wfc)

	for t := range h {
		for i, val := range h[t] {
			r := float32(math.Max(0, float64(val))) //relu
			h[t][i] = r * r                         // Sqaured
		}
	}

	return MatMul(h, wproj) //(T, nEmbed)
}
