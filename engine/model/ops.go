package model

// Add sums two (T, dim) matrices elementwise. Used for residual connections
// where a sublayer's output is added back onto its input.
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
