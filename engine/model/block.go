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
	// The residual is added onto the *unnormalized* x — pre-norm means the
	// residual stream itself is never rescaled, which is why its magnitude
	// grows as layers stack.
	x = Add(x, attnOut)
	p.tr.Stage("+ attention residual", x)

	mlpOut := b.MLP.Forward(b.PostAttnNorm.Forward(x), p.tr)
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
