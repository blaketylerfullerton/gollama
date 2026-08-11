package model

// Block is one transformer layer: pre-norm attention with a residual, then
// pre-norm MLP with a residual. "Pre-norm" means the norm happens on the way
// into each sublayer, so the residual stream itself is never normalized.
type Block struct {
	InputNorm    RMSNorm // before attention
	Attn         Attention
	PostAttnNorm RMSNorm // before the MLP
	MLP          MLP
}

func (b *Block) Forward(x [][]float32, p *pass) [][]float32 {
	normed := b.InputNorm.Forward(x)
	p.tr.Stage("input norm", normed)

	attnOut, weights := b.Attn.Forward(normed, p)
	for h, w := range weights {
		p.tr.Attention(h, w)
	}
	// Both halves of each sublayer are reported: what it produced on its own,
	// and the stream after it landed. Only the second used to be, which made the
	// stream's movement visible but left it unattributable — you could see the
	// magnitude change without seeing what changed it.
	p.tr.Stage("attention out", attnOut)
	// The residual is added onto the *unnormalized* x — pre-norm means the
	// residual stream itself is never rescaled, which is why its magnitude
	// grows as layers stack.
	x = Add(x, attnOut)
	p.tr.Stage("+ attention residual", x)

	mlpOut := b.MLP.Forward(b.PostAttnNorm.Forward(x), p.tr)
	p.tr.Stage("mlp out", mlpOut)
	p.record(ComponentMLP, mlpOut[len(mlpOut)-1])
	x = Add(x, mlpOut)
	p.tr.Stage("+ mlp residual", x)
	return x
}

func NewRandomBlock(cfg GPTConfig) Block {
	return Block{
		InputNorm:    NewRMSNorm(cfg.NEmbed, cfg.NormEps),
		Attn:         NewRandomAttention(cfg),
		PostAttnNorm: NewRMSNorm(cfg.NEmbed, cfg.NormEps),
		MLP:          NewRandomMLP(cfg),
	}
}
