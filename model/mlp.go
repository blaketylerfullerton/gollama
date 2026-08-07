package model

import "math"

// MLP is the feed-forward half of a transformer block, SwiGLU style: two
// parallel projections up to Intermediate width, one gated by SiLU against the
// other, then a single projection back down. Three matrices, not two — that's
// why Qwen3's intermediate size isn't the classic 4*NEmbed.
type MLP struct {
	Gate, Up, Down Linear
}

func (m *MLP) Forward(x [][]float32, tr Trace) [][]float32 {
	gate := MatMul(x, m.Gate) // (T, Intermediate)
	up := MatMul(x, m.Up)     // (T, Intermediate)

	// Gate in place: silu(gate) * up.
	traceOn := tr.On()
	var negative int
	for t := range gate {
		for i, g := range gate[t] {
			if traceOn && g < 0 {
				negative++
			}
			gate[t][i] = silu(g) * up[t][i]
		}
	}
	if traceOn {
		total := len(gate) * len(gate[0])
		tr.Note("SwiGLU gate: %.1f%% of %d units are negative, so SiLU squashes them toward zero",
			100*float64(negative)/float64(total), total)
	}

	return MatMul(gate, m.Down) // (T, NEmbed)
}

// silu (aka swish) is x * sigmoid(x) — smooth, and unlike ReLU it lets a little
// negative signal through instead of hard-zeroing it.
func silu(x float32) float32 {
	return x / float32(1+math.Exp(-float64(x)))
}

func NewRandomMLP(cfg GPTConfig) MLP {
	return MLP{
		Gate: NewRandomLinear(cfg.NEmbed, cfg.Intermediate),
		Up:   NewRandomLinear(cfg.NEmbed, cfg.Intermediate),
		Down: NewRandomLinear(cfg.Intermediate, cfg.NEmbed),
	}
}
