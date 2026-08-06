package model

func (b *Block) Forward(x [][]float32, nHead int) [][]float32 {
	attnOut, _ := MultiHeadAttention(b.LN1.Forward(x), b.Wq, b.Wk, b.Wv, b.Wproj, nHead)
	x = Add(x, attnOut)

	mlpOut := MLP(b.LN2.Forward(x), b.Wfc, b.Wmlp)
	return Add(x, mlpOut)
}

func NewRandomBlock(nEmbd, nHead int) Block {
	headDim := nEmbd / nHead
	return Block{
		LN1:   NewRandomLayerNorm(nEmbd),
		Wq:    NewRandomLinear(nEmbd, nHead*headDim),
		Wk:    NewRandomLinear(nEmbd, nHead*headDim),
		Wv:    NewRandomLinear(nEmbd, nHead*headDim),
		Wproj: NewRandomLinear(nHead*headDim, nEmbd),
		LN2:   NewRandomLayerNorm(nEmbd),
		Wfc:   NewRandomLinear(nEmbd, 4*nEmbd),
		Wmlp:  NewRandomLinear(4*nEmbd, nEmbd),
	}
}
