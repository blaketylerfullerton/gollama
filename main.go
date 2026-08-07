package main

import (
	"fmt"
	"math"
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

	// --- 1. tokenization ----------------------------------------------------
	// Byte-level BPE splits text into subwords, then looks each one up. Printing
	// them one per line is the only way to see where the splits actually land.
	section("1. tokenization")
	fmt.Printf("input: %q\n\n", text)
	fmt.Printf("  %-4s %-8s %s\n", "#", "id", "token")
	labels := make([]string, len(ids))
	for i, id := range ids {
		labels[i] = tok.Decode([]int{id})
		fmt.Printf("  %-4d %-8d %q\n", i, id, labels[i])
	}
	fmt.Printf("\n%d tokens. Round-trip decode: %q\n", len(ids), tok.Decode(ids))

	// --- 2. the model shape -------------------------------------------------
	// A tiny Qwen3-shaped config. Two properties are inherited from the real
	// Qwen3-0.6B on purpose:
	//   HeadDim (16) is NOT NEmbed/NHead (32/4 = 8)
	//   NKVHead (2) < NHead (4), so query heads share kv heads
	// Weights are random, so the predictions are meaningless — the shapes and
	// the mechanics are the point.
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

	section("2. model shape")
	fmt.Printf("%d layers, %d embedding dims, vocab %d\n", cfg.NLayer, cfg.NEmbed, cfg.VocabSize)
	fmt.Printf("%d query heads / %d kv heads x %d head dims\n", cfg.NHead, cfg.NKVHead, cfg.HeadDim)
	fmt.Printf("  q projects to %d wide, k and v only to %d\n", cfg.QOut(), cfg.KVOut())
	fmt.Printf("  grouped-query attention: %d query heads share each kv head,\n", cfg.GroupSize())
	fmt.Printf("  so the KV cache is %dx smaller than it would be otherwise\n", cfg.GroupSize())
	fmt.Printf("  note HeadDim=%d, which is NOT NEmbed/NHead=%d — Qwen3 sets it explicitly\n",
		cfg.HeadDim, cfg.NEmbed/cfg.NHead)
	fmt.Println()
	printParams(cfg)

	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		panic(err)
	}

	// --- 3. the forward pass ------------------------------------------------
	// Everything below this line is printed by the model itself, via the Tracer
	// hook. Forward calls into it at each stage; without a Tracer set it stays
	// completely silent, which is how the tests and any real inference run it.
	trace := &walkthrough{labels: labels, detailLayer: 0, detailHead: 0}
	gpt.Trace = trace

	section("3. forward pass (detail for layer 0, head 0)")
	logits := gpt.Forward(ids)

	// --- 4. positional encoding ---------------------------------------------
	// Built lazily during Forward, so this has to come after it.
	section("4. rotary position tables")
	cos, sin := gpt.RotaryTables()
	PrintRotaryTable(cos, sin, 3)

	// --- 5. summary ---------------------------------------------------------
	section("5. stage magnitudes (mean ‖x‖ over tokens)")
	trace.PrintSummary()

	// --- 6. sampling --------------------------------------------------------
	// Logits are unbounded scores. Softmax turns the last row into a
	// distribution, and temperature decides how peaked that distribution is.
	section("6. next-token distribution")
	last := logits[len(logits)-1]
	fmt.Printf("predicting the token after %q, over a vocab of %d\n", labels[len(labels)-1], cfg.VocabSize)
	for _, temp := range []float64{0.7, 1.0, 1.5} {
		fmt.Printf("\n  temperature %.1f\n", temp)
		fmt.Printf("    %6s  %10s  %14s  %s\n", "id", "p(token)", "share of top 5", "token")
		cands := topK(last, temp, 5)
		var top float64
		for _, c := range cands {
			top += c.prob
		}
		for _, c := range cands {
			fmt.Printf("    %6d  %9.4f%%  %13.1f%%  %q\n",
				c.id, 100*c.prob, 100*c.prob/top, tok.Decode([]int{c.id}))
		}
	}
	fmt.Println("\n  Temperature divides the logits before softmax, so it only changes how")
	fmt.Println("  peaked the distribution is — it can never reorder the candidates.")
	fmt.Printf("  Weights are random here, so all %d tokens sit near 1/%d ≈ %.4f%%.\n",
		cfg.VocabSize, cfg.VocabSize, 100/float64(cfg.VocabSize))
	fmt.Println("  Watch the share-of-top-5 column instead: that is where temperature shows.")

	section("next up")
	fmt.Println("  - a Generate loop that samples, appends, and re-runs")
	fmt.Println("  - load real Qwen3 weights with model.FromDirectory")
	fmt.Println("  - a KV cache so each new token doesn't recompute the whole prefix")
}

func section(title string) {
	fmt.Printf("\n\n=== %s ===\n\n", title)
}

// printParams breaks the parameter count down by where it lives. At small
// scale the embedding table dominates everything else, which surprises people.
func printParams(cfg model.GPTConfig) {
	type group struct {
		name string
		n    int
	}

	attnPerLayer := cfg.NEmbed*cfg.QOut() + // q
		2*cfg.NEmbed*cfg.KVOut() + // k, v
		cfg.QOut()*cfg.NEmbed + // output projection
		2*cfg.HeadDim // q_norm, k_norm
	mlpPerLayer := 3 * cfg.NEmbed * cfg.Intermediate // gate, up, down
	normsPerLayer := 2 * cfg.NEmbed                  // input_layernorm, post_attention_layernorm

	groups := []group{
		{"embeddings", cfg.VocabSize * cfg.NEmbed},
		{"attention (all layers)", cfg.NLayer * attnPerLayer},
		{"mlp (all layers)", cfg.NLayer * mlpPerLayer},
		{"norms", cfg.NLayer*normsPerLayer + cfg.NEmbed},
	}
	if !cfg.TieEmbed {
		groups = append(groups, group{"lm head", cfg.VocabSize * cfg.NEmbed})
	}

	var total int
	for _, g := range groups {
		total += g.n
	}

	fmt.Println("parameters")
	for _, g := range groups {
		fmt.Printf("  %-24s %10d  %5.1f%%\n", g.name, g.n, 100*float64(g.n)/float64(total))
	}
	fmt.Printf("  %-24s %10d\n", "total", total)
	if cfg.TieEmbed {
		fmt.Println("  (embeddings are tied: the lm head reuses this table instead of its own)")
	}
}

type cand struct {
	id   int
	prob float64
}

// topK applies temperature, softmaxes the whole row, and returns the k most
// likely tokens. Dividing by temperature before exponentiating is all
// "temperature" means: it scales the gaps between logits.
func topK(logits []float32, temp float64, k int) []cand {
	maxLogit := float64(logits[0])
	for _, l := range logits {
		if float64(l) > maxLogit {
			maxLogit = float64(l)
		}
	}

	probs := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		// Subtract the max before exp for numerical stability — it cancels out
		// in the division below.
		p := math.Exp((float64(l) - maxLogit) / temp)
		probs[i] = p
		sum += p
	}

	out := make([]cand, len(logits))
	for i, p := range probs {
		out[i] = cand{i, p / sum}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].prob > out[b].prob })

	if k > len(out) {
		k = len(out)
	}
	return out[:k]
}
