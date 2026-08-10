// This is the token embedding step: turning token ids into vectors
// the rest of the transformer can operate on.
package model

// Embed looks up each token id's row in the embedding table.
// wte is flat (vocabSize, nEmbd), row-major. Returns (T, nEmbd) — one
// vector per input token.
func Embed(wte []float32, ids []int, nEmbd int) [][]float32 {
	out := make([][]float32, len(ids))
	for t, id := range ids {
		row := wte[id*nEmbd : (id+1)*nEmbd]
		out[t] = make([]float32, nEmbd)
		copy(out[t], row)
	}
	return out
}

// NewRandomEmbedding creates a randomly initialized (vocabSize, nEmbd)
// embedding table. Useful for testing the embedding step in isolation
// before real trained weights exist.
func NewRandomEmbedding(vocabSize, nEmbd int) []float32 {
	return randFloats(vocabSize * nEmbd)
}
