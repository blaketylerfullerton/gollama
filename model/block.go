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

func (b *Block) Forward(x [][]float32, cos, sin [][]float32) [][]float32 {
	attnOut, _ := b.Attn.Forward(b.InputNorm.Forward(x), cos, sin)
	x = Add(x, attnOut)

	mlpOut := b.MLP.Forward(b.PostAttnNorm.Forward(x))
	return Add(x, mlpOut)
}

func NewRandomBlock(cfg GPTConfig) Block {
	return Block{
		InputNorm:    NewRMSNorm(cfg.NEmbed, cfg.NormEps),
		Attn:         NewRandomAttention(cfg),
		PostAttnNorm: NewRMSNorm(cfg.NEmbed, cfg.NormEps),
		MLP:          NewRandomMLP(cfg),
	}
}
