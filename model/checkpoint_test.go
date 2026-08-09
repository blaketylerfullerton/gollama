package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/blaketylerfullerton/GoLlama/tokenizer"
)

// checkpointDir is where the Qwen3-0.6B download lands. These are integration
// tests: they skip when the checkpoint is absent so a fresh clone still passes,
// and skip under -short because they read 1.5GB and take seconds, not
// microseconds.
const checkpointDir = "../checkpoints/qwen3-0.6b"

func realGPT(t *testing.T) *GPT {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping checkpoint test in -short mode")
	}
	if _, err := os.Stat(filepath.Join(checkpointDir, "model.safetensors")); err != nil {
		t.Skipf("no checkpoint at %s — run: huggingface-cli download Qwen/Qwen3-0.6B --local-dir %s",
			checkpointDir, checkpointDir)
	}
	gpt, err := FromDirectory(checkpointDir)
	if err != nil {
		t.Fatalf("loading the real checkpoint: %v", err)
	}
	return gpt
}

// st opens the checkpoint's safetensors directly, for assertions about what the
// file contains rather than what the loader made of it.
func st(t *testing.T) *Safetensors {
	t.Helper()
	s, err := OpenSafetensorsDir(checkpointDir)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestRealCheckpointConfig(t *testing.T) {
	gpt := realGPT(t)
	c := gpt.Config

	// Pinning these guards against a config loader regression going unnoticed.
	// HeadDim is the important one: 128, while NEmbed/NHead is 1024/16 = 64.
	for _, tc := range []struct {
		name string
		got  int
		want int
	}{
		{"NLayer", c.NLayer, 28},
		{"NEmbed", c.NEmbed, 1024},
		{"NHead", c.NHead, 16},
		{"NKVHead", c.NKVHead, 8},
		{"HeadDim", c.HeadDim, 128},
		{"Intermediate", c.Intermediate, 3072},
		{"VocabSize", c.VocabSize, 151936},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.HeadDim == c.NEmbed/c.NHead {
		t.Error("HeadDim happens to equal NEmbed/NHead, so this checkpoint no longer " +
			"exercises the explicit-head_dim path")
	}
	if c.RopeBase != 1e6 {
		t.Errorf("RopeBase = %v, want 1e6", c.RopeBase)
	}
	if !c.TieEmbed {
		t.Error("Qwen3-0.6B ties embeddings")
	}

	// generation_config.json lists both stop tokens; config.json lists only one.
	if len(c.EOSTokenIDs) != 2 {
		t.Errorf("EOSTokenIDs = %v, want two ids from generation_config.json", c.EOSTokenIDs)
	}

	if got, want := len(gpt.WTE), c.VocabSize*c.NEmbed; got != want {
		t.Errorf("embedding table has %d floats, want %d", got, want)
	}
	// This checkpoint declares tie_word_embeddings AND ships an identical
	// lm_head.weight. Aliasing rather than loading it saves 622MB of float32.
	if !st(t).Has("lm_head.weight") {
		t.Log("note: checkpoint no longer ships a redundant lm_head.weight")
	}
	if &gpt.LMHead.Weight[0] != &gpt.WTE[0] {
		t.Error("tied lm_head should alias the embedding table, not load a second copy")
	}
}

func TestRealCheckpointGQAShapes(t *testing.T) {
	gpt := realGPT(t)
	c := gpt.Config
	a := gpt.Blocks[0].Attn

	if a.Wq.Out != 2048 || a.Wq.In != 1024 {
		t.Errorf("Wq is %dx%d, want 2048x1024", a.Wq.Out, a.Wq.In)
	}
	// Half as wide as q: that asymmetry is the whole point of GQA.
	if a.Wk.Out != 1024 || a.Wv.Out != 1024 {
		t.Errorf("Wk/Wv out = %d/%d, want 1024 each", a.Wk.Out, a.Wv.Out)
	}
	if len(a.QNorm.Weight) != c.HeadDim || len(a.KNorm.Weight) != c.HeadDim {
		t.Errorf("QK-norm weights are %d/%d long, want %d",
			len(a.QNorm.Weight), len(a.KNorm.Weight), c.HeadDim)
	}
}

// The end-to-end correctness test. A real model asked for the capital of France
// answers confidently; anything wrong in the attention math — rotary sign
// convention, QK-norm ordering, GQA head mapping, a transposed weight — turns
// this into a flat distribution over nonsense rather than an error.
func TestRealCheckpointPredictsParis(t *testing.T) {
	gpt := realGPT(t)
	tok, err := tokenizer.FromDirectory(checkpointDir)
	if err != nil {
		t.Fatal(err)
	}

	// "The capital of France is" — hardcoded because Encode is still
	// approximate. The round-trip below is what proves the ids are right.
	ids := []int{785, 6722, 315, 9625, 374}
	if got, want := tok.Decode(ids), "The capital of France is"; got != want {
		t.Fatalf("prompt ids decode to %q, want %q", got, want)
	}

	logits := gpt.Forward(ids)
	if len(logits) != len(ids) || len(logits[0]) != gpt.Config.VocabSize {
		t.Fatalf("logits are %dx%d, want %dx%d",
			len(logits), len(logits[0]), len(ids), gpt.Config.VocabSize)
	}
	for _, row := range logits {
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("logit %d is %v", i, v)
			}
		}
	}

	top := TopCandidates(logits[len(logits)-1], 1.0, 5)
	if got := tok.Decode([]int{top[0].ID}); got != " Paris" {
		t.Errorf("top prediction is %q, want %q", got, " Paris")
		for i, c := range top {
			t.Logf("  %d. %6.2f%% %q", i+1, 100*c.Prob, tok.Decode([]int{c.ID}))
		}
	}
	// Confidence matters as much as ranking: a subtly broken forward pass can
	// still rank Paris first while spreading mass almost uniformly.
	if top[0].Prob < 0.4 {
		t.Errorf("top probability is %.1f%%, want > 40%% — the distribution is too flat "+
			"for a model that knows this fact", 100*top[0].Prob)
	}
}

func TestRealCheckpointGreedyIsSensible(t *testing.T) {
	gpt := realGPT(t)
	tok, err := tokenizer.FromDirectory(checkpointDir)
	if err != nil {
		t.Fatal(err)
	}

	// Kept short deliberately: without a KV cache each step recomputes the
	// whole prefix, so this is seconds per token.
	ids := []int{16, 220, 17, 220, 18, 220, 19, 220, 20} // "1 2 3 4 5"
	out, err := gpt.Generate(ids, GenerateOpts{MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	got := tok.DecodeSkipSpecial(out)
	if got != " 6" {
		t.Errorf("continuing %q gave %q, want %q", tok.Decode(ids), got, " 6")
	}
}

// The equivalence check that matters, on the real model rather than a tiny
// synthetic one. 28 layers and 40960 rotary positions give a wrong position
// offset far more room to show up than the tiny config does.
func TestRealCheckpointCachedMatchesUncached(t *testing.T) {
	gpt := realGPT(t)
	prompt := []int{785, 6722, 315, 9625, 374} // "The capital of France is"

	cached, err := gpt.Generate(prompt, GenerateOpts{MaxTokens: 6})
	if err != nil {
		t.Fatal(err)
	}
	uncached, err := gpt.Generate(prompt, GenerateOpts{MaxTokens: 6, NoCache: true})
	if err != nil {
		t.Fatal(err)
	}

	if len(cached) != len(uncached) {
		t.Fatalf("cached produced %d tokens, uncached %d", len(cached), len(uncached))
	}
	for i := range cached {
		if cached[i] != uncached[i] {
			tok, _ := tokenizer.FromDirectory(checkpointDir)
			t.Fatalf("diverged at token %d: cached %q, uncached %q",
				i, tok.Decode(cached), tok.Decode(uncached))
		}
	}
}

func TestRealCheckpointCacheSize(t *testing.T) {
	gpt := realGPT(t)
	cache := NewKVCache(gpt.Config)

	// 28 layers x 8 kv heads x 128 dims x 2 tensors x 4 bytes = 229376.
	if got, want := cache.BytesPerToken(), 28*8*128*2*4; got != want {
		t.Errorf("BytesPerToken is %d, want %d", got, want)
	}
	// Without GQA this would be NHead wide instead of NKVHead — twice as big.
	withoutGQA := 28 * gpt.Config.NHead * 128 * 2 * 4
	if withoutGQA != cache.BytesPerToken()*gpt.Config.GroupSize() {
		t.Errorf("GQA saving is not the expected %dx", gpt.Config.GroupSize())
	}
}

// BenchmarkRealForward measures a full uncached forward over the prompt.
// Run with: go test ./model/ -bench . -benchtime 3x
func BenchmarkRealForward(b *testing.B) {
	if _, err := os.Stat(filepath.Join(checkpointDir, "model.safetensors")); err != nil {
		b.Skipf("no checkpoint at %s", checkpointDir)
	}
	gpt, err := FromDirectory(checkpointDir)
	if err != nil {
		b.Fatal(err)
	}
	ids := []int{785, 6722, 315, 9625, 374}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gpt.Forward(ids)
	}
}

// BenchmarkRealDecode measures one cached decode step against a warm cache —
// the operation that dominates generation.
func BenchmarkRealDecode(b *testing.B) {
	if _, err := os.Stat(filepath.Join(checkpointDir, "model.safetensors")); err != nil {
		b.Skipf("no checkpoint at %s", checkpointDir)
	}
	gpt, err := FromDirectory(checkpointDir)
	if err != nil {
		b.Fatal(err)
	}
	cache := NewKVCache(gpt.Config)
	if _, err := gpt.ForwardCached([]int{785, 6722, 315, 9625, 374}, cache); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := gpt.ForwardCached([]int{374}, cache); err != nil {
			b.Fatal(err)
		}
	}
}
