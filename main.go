package main

import (
	"fmt"
	"reflect"

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

	fmt.Println("Embedding shape:", len(vectors), "tokens x", nEmbd, "dims")
	fmt.Println(colHeader(previewDims))
	fmt.Println(vecRow("tok 0", vectors[0], previewDims))
	fmt.Println("---------------")

	headDim := nEmbd //pretending the whole embedding vector is one head for this demo

	wq := model.NewRandomLinear(nEmbd, headDim)
	wk := model.NewRandomLinear(nEmbd, headDim)
	wv := model.NewRandomLinear(nEmbd, headDim)

	q := model.MatMul(vectors, wq) // (T, headDim)
	k := model.MatMul(vectors, wk) // (T, headDim)
	v := model.MatMul(vectors, wv)

	cos, sin := model.PrecomputeRotary(len(ids), headDim, 10000)

	for t := range q {
		q[t] = model.ApplyRotary(q[t], cos[t], sin[t])
		k[t] = model.ApplyRotary(k[t], cos[t], sin[t])

		//QK Norm
		q[t] = model.RMSNormVec(q[t])
		k[t] = model.RMSNormVec(k[t])
	}

	//Causal Attention (Scores, softmax, weigthted sum over V)
	attnOut, weights := model.CausalAttention(q, k, v) // (T, headDim)

	fmt.Println("Attention out of shape: ", len(attnOut), "x", len(attnOut[0]))
	fmt.Println("attnOut[0] == v[0]: ", reflect.DeepEqual(attnOut[0], v))
	fmt.Println(weights)

	//Output projection
	wproj := model.NewRandomLinear(headDim, nEmbd)
	attnOut = model.MatMul(attnOut, wproj)

}
