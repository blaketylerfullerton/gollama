package model

func (b *Block) Forward(x [][]float32, cos, sin [][]float32, nHead int) [][]float32 {
	attnIn := RMSNorm(x)
	attnOut, _ := MultiHeadAttention(attnIn, b.Wq, b.Wk, b.Wv, b.Wproj, cos, sin, nHead)
	x = Add(x, attnOut)

	mlpOut := MLP(RMSNorm(x), b.Wfc, b.Wmlp)
	return Add(x, mlpOut)
}

func NewRandomBlock(nEmbd, nHead int) Block {
	headDim := nEmbd / nHead
	return Block{
		Wq:    NewRandomLinear(nEmbd, nHead*headDim),
		Wk:    NewRandomLinear(nEmbd, nHead*headDim),
		Wv:    NewRandomLinear(nEmbd, nHead*headDim),
		Wproj: NewRandomLinear(nHead*headDim, nEmbd),
		Wfc:   NewRandomLinear(nEmbd, 4*nEmbd),
		Wmlp:  NewRandomLinear(4*nEmbd, nEmbd),
	}
}
