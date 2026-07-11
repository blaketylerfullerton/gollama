package model

import "math"

// embed: look up each token id's row in the embedding table
// wte is flat (vocab_size, n_embd), returns (T, n_embd)
func embed(wte []float32, ids []int, nEmbd int) [][]float32 {
	out := make([][]float32, len(ids))
	for t, id := range ids {
		row := wte[id*nEmbd : (id+1)*nEmbd]
		out[t] = make([]float32, nEmbd)
		copy(out[t], row)
	}
	return out
}
func add(a, b [][]float32) [][]float32 {
	out := make([][]float32, len(a))
	for i := range a {
		out[i] = make([]float32, len(a[i]))
		for j := range a[i] {
			out[i][j] = a[i][j] + b[i][j]
		}
	}
	return out
}

func softcap(x [][]float32, cap float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, row := range x {
		out[i] = make([]float32, len(row))
		for j, v := range row {
			out[i][j] = cap * float32(math.Tanh(float64(v/cap)))
		}
	}
	return out
}

// matmul: x is (T, in), computes x @ W^T using the Linear's weight (out, in)
// returns (T, out)
func matmul(x [][]float32, lin Linear) [][]float32 {
	out := make([][]float32, len(x))
	for t, row := range x {
		out[t] = make([]float32, lin.Out)
		for o := 0; o < lin.Out; o++ {
			var sum float32
			weightRow := lin.Weight[o*lin.In : (o+1)*lin.In] // row o of the (out, in) weight
			for i := 0; i < lin.In; i++ {
				sum += row[i] * weightRow[i]
			}
			out[t][o] = sum
		}
	}
	return out
}

// applyRotary: split last dim in half, rotate pairs
func applyRotary(x []float32, cos, sin []float32) []float32 {
	d := len(x) / 2
	out := make([]float32, len(x))
	for i := 0; i < d; i++ {
		x1, x2 := x[i], x[i+d]
		out[i] = x1*cos[i] + x2*sin[i]
		out[i+d] = -x1*sin[i] + x2*cos[i]
	}
	return out
}
func rmsNormVec(x []float32) []float32 {
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

func rmsNorm(x [][]float32) [][]float32 {
	out := make([][]float32, len(x))
	for i, row := range x {
		out[i] = rmsNormVec(row)
	}
	return out
}

// mlp implements MLP.forward from gpt.py: c_fc -> relu().square() -> c_proj
func (b *Block) mlp(x [][]float32) [][]float32 {
	h := matmul(x, b.Wfc) // (T, 4*n_embd)

	for t := range h {
		for i, v := range h[t] {
			r := float32(math.Max(0, float64(v))) // relu
			h[t][i] = r * r                       // squared
		}
	}

	return matmul(h, b.Wmlp) // (T, n_embd)
}

// precomputeRotary computes cos/sin tables matching _precompute_rotary_embeddings in gpt.py.
// Returns two (seqLen, headDim/2) matrices — cos[t] and sin[t] give the rotation for position t.
func precomputeRotary(seqLen, headDim int, base float64) (cos, sin [][]float32) {
	halfDim := headDim / 2

	// inv_freq[i] = 1 / base^(2i/headDim), for i in 0..halfDim
	invFreq := make([]float64, halfDim)
	for i := 0; i < halfDim; i++ {
		channel := float64(2 * i) // matches torch.arange(0, head_dim, 2)
		invFreq[i] = 1.0 / math.Pow(base, channel/float64(headDim))
	}

	cos = make([][]float32, seqLen)
	sin = make([][]float32, seqLen)
	for t := 0; t < seqLen; t++ {
		cos[t] = make([]float32, halfDim)
		sin[t] = make([]float32, halfDim)
		for i := 0; i < halfDim; i++ {
			freq := float64(t) * invFreq[i] // torch.outer(t, inv_freq)
			cos[t][i] = float32(math.Cos(freq))
			sin[t][i] = float32(math.Sin(freq))
		}
	}
	return cos, sin
}
