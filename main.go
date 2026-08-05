package main

import (
	"fmt"

	"github.com/blaketylerfullerton/GoLlama/model"
	"github.com/blaketylerfullerton/GoLlama/tokenizer"
)

func main() {
	tok, err := tokenizer.FromDirectory("tokenizer/testdata")
	if err != nil {
		panic(err)
	}

	text := "Hello, world. Testing tokenization layer"
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)

	fmt.Println("Input:  ", text)
	fmt.Println("Ids:    ", ids)
	fmt.Println("Decoded:", decoded)
	fmt.Println("---------------")

	// Embedding step: turn each token id into a vector.
	// We don't have real trained weights yet, so we use a randomly
	// initialized table sized to the tokenizer's vocab — this just lets
	// us see the embedding lookup working end to end.
	nEmbd := 32
	wte := model.NewRandomEmbedding(tok.VocabSize(), nEmbd)
	vectors := model.Embed(wte, ids, nEmbd)
	vectors = model.RMSNorm(vectors)

	fmt.Println("Embedding shape:", len(vectors), "tokens x", nEmbd, "dims")
	fmt.Println(colHeader(previewDims))
	fmt.Println(vecRow("tok 0", vectors[0], previewDims))
	fmt.Println("---------------")

	nHead := 4
	headDim := nEmbd / nHead

	wq := model.NewRandomLinear(nEmbd, nHead*headDim)
	wk := model.NewRandomLinear(nEmbd, nHead*headDim)
	wv := model.NewRandomLinear(nEmbd, nHead*headDim)
	wproj := model.NewRandomLinear(nHead*headDim, nEmbd)

	cos, sin := model.PrecomputeRotary(len(ids), headDim, 10000)

	attnIn := model.RMSNorm(vectors) // seperate normed copy, input to attention only

	attnOut, _ := model.MultiHeadAttention(attnIn, wq, wk, wv, wproj, cos, sin, nHead)
	x := model.Add(vectors, attnOut)


	//MLP (Multi Layer perceptron)
	normed := model.RMSNorm(x)
	wfc := model.NewRandomLinear(nEmbd, 4*nEmbd)
	wmlpProj := model.NewRandomLinear(4*nEmbd, nEmbd)
	mlpOut := model.MLP(normed, wfc, wmlpProj)
	x = model.Add(x, mlpOut) //second residual. Block Complete

	// Final Norm + LM head
	x = model.RMSNorm(x)
	lmHead := model.NewRandomLinear(nEmbd, tok.VocabSize())
	logits := model.MatMul(x, lmHead)
	logits = model.SoftCap(logits, 15) // 15*tanh(x/15)

	fmt.Println("---------------")
	last := logits[len(logits)-1]
	fmt.Println("Logits shape:", len(logits), "x", len(last))

}
