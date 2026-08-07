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

	fmt.Println("Input:  ", text)
	fmt.Println("Ids:    ", ids)
	fmt.Println("Decoded:", tok.Decode(ids))
	fmt.Println("---------------")

	// A tiny Qwen3-shaped config. The two things worth noticing:
	//   HeadDim (16) is NOT NEmbed/NHead (32/4 = 8)
	//   NKVHead (2) < NHead (4), so query heads share kv heads
	// Both mirror the real Qwen3-0.6B, so the shapes exercised here are the
	// same ones a real checkpoint will hit. Weights are still random.
	cfg := model.GPTConfig{
		VocabSize:    tok.VocabSize(),
		NLayer:       2,
		NHead:        4,
		NKVHead:      2,
		NEmbed:       32,
		HeadDim:      16,
		Intermediate: 96,
		RopeBase:     1e6,
		NormEps:      1e-6,
		TieEmbed:     true,
		SequenceLen:  512,
	}

	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Model: %d layers, %d embd, %d q heads / %d kv heads x %d head dim\n",
		cfg.NLayer, cfg.NEmbed, cfg.NHead, cfg.NKVHead, cfg.HeadDim)
	fmt.Printf("       q proj %d wide, kv proj %d wide (group size %d)\n",
		cfg.QOut(), cfg.KVOut(), cfg.GroupSize())
	fmt.Println("---------------")

	logits := gpt.Forward(ids)

	last := logits[len(logits)-1]
	fmt.Println("Logits shape:", len(logits), "x", len(last))
	fmt.Println(colHeader(previewDims))
	fmt.Println(vecRow("logits", last, previewDims))
	fmt.Println("---------------")

	type cand struct {
		id    int
		logit float32
	}
	cands := make([]cand, len(last))
	for i, l := range last {
		cands[i] = cand{i, l}
	}
	sort.Slice(cands, func(a, b int) bool { return cands[a].logit > cands[b].logit })

	fmt.Println("Top 5 next-token predictions:")
	for _, c := range cands[:5] {
		fmt.Printf(" %6d %8.4f %q\n", c.id, c.logit, tok.Decode([]int{c.id}))
	}
}
