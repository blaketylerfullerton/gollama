// This is the Transformer block and forward pass
package model

import (
	"math/rand/v2"
)

type GPTConfig struct {
	SequenceLen int
	VocabSize   int
	NLayer      int
	NHead       int
	NKVHead     int
	NEmbed      int
	Rotary      float64 //1000 at this commit dont hardcode it
}

type Linear struct {
	Weight  []float32 // shape (out, in), row-major, no bias
	In, Out int
}

type Block struct {
	Wq, Wk, Wv, Wproj Linear // attention
	Wfc, Wmlp         Linear // mlp (c_fc, c_proj)
}

type GPT struct {
	Config   GPTConfig
	WTE      []float32
	Blocks   []Block
	LMHead   Linear
	Cos, Sin [][]float32
}

func randFloats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rand.NormFloat64() * 0.02)
	}
	return out
}
