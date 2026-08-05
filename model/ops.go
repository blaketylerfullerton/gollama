package model

import "math"

// Add sums two (T, dim) matrices elementwise. Used for residual Connections
// where the blocks output is added back onto its input
func Add(a, b [][]float32) [][]float32 {
	out := make([][]float32, len(a))
	for i := range a {
		out[i] = make([]float32, len(a[i]))
		for j := range a[i] {
			out[i][j] = a[i][j] + b[i][j]
		}
	}
	return out
}

func SoftCap(x [][]float32, limit float32) [][]float32 {
	out := make([][]float32, len(x))
	for i , row := range x {
		out[i] = make([]float32, len(row))
		for j, v := range row {
			out[i][j] = limit * float32(math.Tanh(float64(v/limit)))
		}
	}
	return out
}
