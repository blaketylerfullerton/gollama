package model

// NewRandomLinear creates a randomly initialized Linear layer with no bias.
// Weight is (out, in) row-major. Qwen3's attention and MLP projections are all
// bias-free, so Bias stays nil — MatMul skips it.
func NewRandomLinear(in, out int) Linear {
	return Linear{
		Weight: randFloats(in * out),
		Bias:   nil,
		In:     in,
		Out:    out,
	}
}

// MatMul projects x (T, in) through a Linear layer's weights, computing
// x @ W^T. Returns (T, out).
func MatMul(x [][]float32, lin Linear) [][]float32 {
	out := make([][]float32, len(x))
	for t, row := range x {
		out[t] = make([]float32, lin.Out)
		for o := 0; o < lin.Out; o++ {
			var sum float32
			weightRow := lin.Weight[o*lin.In : (o+1)*lin.In]
			for i := 0; i < lin.In; i++ {
				sum += row[i] * weightRow[i]
			}
			if lin.Bias != nil {
				sum += lin.Bias[o]
			}
			out[t][o] = sum
		}
	}
	return out
}
