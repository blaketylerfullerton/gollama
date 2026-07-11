// This is the Transformer block and forward pass
package model

import (
	"math"
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

func (m *GPT) Forward(ids []int) [][]float32 {
	T := len(ids)
	headDim := m.Config.NEmbed / m.Config.NHead

	x := embed(m.WTE, ids, m.Config.NEmbed) // (T, embed)
	x = rmsNorm(x)                          // Norm after embedding

	for _, block := range m.Blocks {
		normed := rmsNorm(x)
		attnOut := block.attention(normed, m.Cos, m.Sin, headDim, m.Config.NHead, m.Config.NKVHead, T)

		x = add(x, attnOut)

		normed = rmsNorm(x)
		mlpOut := block.mlp(normed)
		x = add(x, mlpOut)
	}

	x = rmsNorm(x)
	logits := matmul(x, m.LMHead)  // (T, vocab_size)
	logits = softcap(logits, 15.0) // 15*tanh(logits/15)
	return logits

}

// attention implements CausalSelfAttention.forward from gpt.py.
// x is (T, n_embd). cos/sin are already sliced to the current T positions,
// each row of length headDim/2.
func (b *Block) attention(x [][]float32, cos, sin [][]float32, headDim, nHead, nKVHead, T int) [][]float32 {
	q := matmul(x, b.Wq) // (T, nHead*headDim)
	k := matmul(x, b.Wk) // (T, nKVHead*headDim)
	v := matmul(x, b.Wv) // (T, nKVHead*headDim) — no rotary/norm applied to v

	// Apply rotary embedding, then QK-norm, to every head of q and k.
	// Order matters: rotary first, THEN norm — matches apply_rotary_emb(...) followed by norm(q), norm(k) in Python.
	for t := 0; t < T; t++ {
		for h := 0; h < nHead; h++ {
			start := h * headDim
			head := q[t][start : start+headDim]
			copy(head, rmsNormVec(applyRotary(head, cos[t], sin[t])))
		}
		for h := 0; h < nKVHead; h++ {
			start := h * headDim
			head := k[t][start : start+headDim]
			copy(head, rmsNormVec(applyRotary(head, cos[t], sin[t])))
		}
	}

	nRep := nHead / nKVHead // MQA: how many query heads share each kv head
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	out := make([][]float32, T)
	for t := range out {
		out[t] = make([]float32, nHead*headDim)
	}

	// For each query head, causal attention over the (possibly shared) kv head.
	for h := 0; h < nHead; h++ {
		kvHead := h / nRep // which kv head this query head reads from

		for tq := 0; tq < T; tq++ {
			qVec := q[tq][h*headDim : (h+1)*headDim]

			// Causal: only score against keys at positions 0..tq
			scores := make([]float32, tq+1)
			maxScore := float32(math.Inf(-1))
			for tk := 0; tk <= tq; tk++ {
				kVec := k[tk][kvHead*headDim : (kvHead+1)*headDim]
				var dot float32
				for i := 0; i < headDim; i++ {
					dot += qVec[i] * kVec[i]
				}
				dot *= scale
				scores[tk] = dot
				if dot > maxScore {
					maxScore = dot
				}
			}

			// Softmax (subtract max first for numerical stability)
			var sumExp float32
			for i, s := range scores {
				scores[i] = float32(math.Exp(float64(s - maxScore)))
				sumExp += scores[i]
			}
			for i := range scores {
				scores[i] /= sumExp
			}

			// Weighted sum of values into this (t, h) output slot
			outVec := out[tq][h*headDim : (h+1)*headDim]
			for tk := 0; tk <= tq; tk++ {
				vVec := v[tk][kvHead*headDim : (kvHead+1)*headDim]
				w := scores[tk]
				for i := 0; i < headDim; i++ {
					outVec[i] += w * vVec[i]
				}
			}
		}
	}

	return matmul(out, b.Wproj)
}

func NewRandomGPT(cfg GPTConfig) *GPT {
	headDim := cfg.NEmbed / cfg.NHead
	g := &GPT{Config: cfg}

	g.WTE = randFloats(cfg.VocabSize * cfg.NEmbed)
	g.LMHead = Linear{Weight: randFloats(cfg.VocabSize * cfg.NEmbed), In: cfg.NEmbed, Out: cfg.VocabSize}

	for i := 0; i < cfg.NLayer; i++ {
		g.Blocks = append(g.Blocks, Block{
			Wq:    Linear{Weight: randFloats(cfg.NEmbed * cfg.NHead * headDim), In: cfg.NEmbed, Out: cfg.NHead * headDim},
			Wk:    Linear{Weight: randFloats(cfg.NEmbed * cfg.NKVHead * headDim), In: cfg.NEmbed, Out: cfg.NKVHead * headDim},
			Wv:    Linear{Weight: randFloats(cfg.NEmbed * cfg.NKVHead * headDim), In: cfg.NEmbed, Out: cfg.NKVHead * headDim},
			Wproj: Linear{Weight: randFloats(cfg.NEmbed * cfg.NEmbed), In: cfg.NEmbed, Out: cfg.NEmbed},
			Wfc:   Linear{Weight: randFloats(cfg.NEmbed * 4 * cfg.NEmbed), In: cfg.NEmbed, Out: 4 * cfg.NEmbed},
			Wmlp:  Linear{Weight: randFloats(4 * cfg.NEmbed * cfg.NEmbed), In: 4 * cfg.NEmbed, Out: cfg.NEmbed},
		})
	}

	g.Cos, g.Sin = precomputeRotary(cfg.SequenceLen, headDim, cfg.Rotary)
	return g
}

func randFloats(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rand.NormFloat64() * 0.02)
	}
	return out
}
