package model

// dotProduct computes a . b using four independent accumulators instead of
// one. A single accumulator (sum += a[i]*b[i]) makes every iteration wait on
// the previous one's result, so the CPU's multiply-add pipelines sit mostly
// idle; four independent chains break that dependency and let them overlap.
func dotProduct(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	sum := s0 + s1 + s2 + s3
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

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
