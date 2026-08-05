package main

import (
	"fmt"
	"sort"

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
	nLayer := 2
	headDim := nEmbd / nHead

	cos, sin := model.PrecomputeRotary(len(ids), headDim, 10000)

	blocks := make([]model.Block, nLayer)
	for i := range blocks {
		blocks[i] = model.NewRandomBlock(nEmbd, nHead)
	}

	x := vectors
	for i := range blocks {
		x = blocks[i].Forward(x, cos, sin, nHead)
	}

	// Final Norm + LM head
	x = model.RMSNorm(x)
	lmHead := model.NewRandomLinear(nEmbd, tok.VocabSize())
	logits := model.MatMul(x, lmHead)
	logits = model.SoftCap(logits, 15) // 15*tanh(x/15)

	fmt.Println("---------------")
	last := logits[len(logits)-1]
	fmt.Println("Logits shape:", len(logits), "x", len(last))

	type cand struct {
		id    int
		logit float32
	}
	cands := make([]cand, len(last))
	for i, l := range last {
		cands[i] = cand{i, l}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].logit > cands[b].logit })

	fmt.Println("Top 5 next-token predictions: ")
	for _, c := range cands[:5] {
		fmt.Printf(" %6d %8.4f %q\n", c.id, c.logit, tok.Decode([]int{c.id}))
	}

}
