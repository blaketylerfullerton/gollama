package model

// New Random Linear creates a random intialized Linear Layer
// Weight is (out, in) row major, matching reference of GPT-2
func NewRandomLinear(in, out int) Linear {
	return Linear{
		Weight: randFloats(in * out),
		In:     in,
		Out:    out,
	}
}

// MatMul projects x (T, in) through a Linear layers weights, computing
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
			out[t][o] = sum
		}
	}
	return out
}
