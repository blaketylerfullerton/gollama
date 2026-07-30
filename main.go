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

	text := "Hello, world! This is GoLlama."
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)

	fmt.Println("Input:  ", text)
	fmt.Println("Ids:    ", ids)
	fmt.Println("Decoded:", decoded)
	fmt.Println()

	cfg := model.GPTConfig{
		SequenceLen: 128,
		VocabSize:   1000, // small fake vocab for testing
		NLayer:      2,
		NHead:       4,
		NKVHead:     4,
		NEmbed:      32,
		Rotary:      10000,
	}

	gpt := model.NewRandomGPT(cfg) // helper you'll add — fills weights with random values

	tokens := []int{5, 10, 15, 20}
	logits := gpt.Forward(tokens)

	fmt.Printf("Output shape: %d tokens x %d vocab\n", len(logits), len(logits[0]))
	fmt.Println("First token's top-5 logits:", logits[0][:5])
}
