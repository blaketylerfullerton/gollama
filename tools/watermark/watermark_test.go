package watermark

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
)

func tinyGPT(t *testing.T) *model.GPT {
	t.Helper()
	cfg := model.GPTConfig{
		VocabSize:    64,
		NLayer:       1,
		NHead:        2,
		NKVHead:      2,
		NEmbed:       8,
		HeadDim:      4,
		Intermediate: 16,
		RopeBase:     10000,
		NormEps:      1e-5,
		TieEmbed:     true,
		SequenceLen:  256,
	}
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return gpt
}

func TestContextSeedDeterministic(t *testing.T) {
	a := contextSeed(1, []int{1, 2, 3})
	b := contextSeed(1, []int{1, 2, 3})
	if a != b {
		t.Errorf("same key and context produced different seeds: %d vs %d", a, b)
	}
}

func TestContextSeedSensitiveToKeyAndContext(t *testing.T) {
	base := contextSeed(1, []int{1, 2, 3})
	if got := contextSeed(2, []int{1, 2, 3}); got == base {
		t.Error("different key produced the same seed")
	}
	if got := contextSeed(1, []int{1, 2, 4}); got == base {
		t.Error("different context produced the same seed")
	}
	if got := contextSeed(1, []int{3, 2, 1}); got == base {
		t.Error("reordered context produced the same seed")
	}
}

func TestGValueRange(t *testing.T) {
	for layer := 0; layer < 4; layer++ {
		for token := 0; token < 200; token++ {
			g := gValue(42, layer, token)
			if g < 0 || g >= 1 {
				t.Fatalf("gValue(42, %d, %d) = %v, want [0,1)", layer, token, g)
			}
		}
	}
}

func TestGValueMeanNearOneHalf(t *testing.T) {
	// Not a proof of uniformity, just a sanity check that the mix doesn't
	// obviously skew toward one end — a real bug here would show up as a
	// mean nowhere near 0.5 over enough samples.
	var sum float64
	const n = 20000
	for i := 0; i < n; i++ {
		sum += gValue(uint64(i), i%4, i*7919)
	}
	mean := sum / n
	if math.Abs(mean-0.5) > 0.02 {
		t.Errorf("mean g-value over %d samples = %v, want close to 0.5", n, mean)
	}
}

func TestGenerateRespectsMaxTokens(t *testing.T) {
	gpt := tinyGPT(t)
	cfg := Config{Key: 7, ContextSize: 2, Layers: 2}
	for _, n := range []int{0, 1, 5} {
		got, err := Generate(gpt, cfg, []int{1, 2}, GenerateOpts{MaxTokens: n, Seed: 1})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != n {
			t.Errorf("MaxTokens %d: got %d tokens", n, len(got))
		}
	}
}

func TestGenerateIsDeterministicUnderSeed(t *testing.T) {
	gpt := tinyGPT(t)
	cfg := Config{Key: 7, ContextSize: 3, Layers: 3}
	opts := GenerateOpts{MaxTokens: 10, Temperature: 1, Seed: 5}

	a, err := Generate(gpt, cfg, []int{3, 1}, opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Generate(gpt, cfg, []int{3, 1}, opts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed diverged at token %d: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestGenerateRejectsBadInput(t *testing.T) {
	gpt := tinyGPT(t)
	cfg := Config{Key: 1, ContextSize: 2, Layers: 2}

	if _, err := Generate(gpt, cfg, nil, GenerateOpts{MaxTokens: 4}); err == nil {
		t.Error("expected an error for an empty prompt")
	}
	if _, err := Generate(gpt, cfg, []int{1}, GenerateOpts{MaxTokens: -1}); err == nil {
		t.Error("expected an error for negative MaxTokens")
	}
}

// TestDetectDistinguishesWatermarkedText is the demo's actual claim: text
// produced by Generate scores far above 0.5 under Detect with the same
// Config, while text with no relationship to the key scores near 0.5.
func TestDetectDistinguishesWatermarkedText(t *testing.T) {
	gpt := tinyGPT(t)
	cfg := Config{Key: 0xC0FFEE, ContextSize: 4, Layers: 4}
	prompt := []int{1, 2, 3, 4, 5}

	out, err := Generate(gpt, cfg, prompt, GenerateOpts{MaxTokens: 200, Temperature: 1, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	full := append(append([]int{}, prompt...), out...)
	watermarked := Detect(cfg, full)

	rng := rand.New(rand.NewPCG(99, 99))
	random := make([]int, len(full))
	for i := range random {
		random[i] = rng.IntN(gpt.Config.VocabSize)
	}
	plain := Detect(cfg, random)

	if watermarked.Z <= plain.Z {
		t.Fatalf("watermarked z-score (%v) should exceed unrelated text's (%v)",
			watermarked.Z, plain.Z)
	}
	if watermarked.Z < 4 {
		t.Errorf("watermarked z-score = %v, want well above the ~4 'detected' line", watermarked.Z)
	}
	if math.Abs(plain.Z) > 4 {
		t.Errorf("unrelated text's z-score = %v, want close to 0", plain.Z)
	}
}

// TestDetectHandlesShortSequences makes sure a sequence shorter than
// ContextSize doesn't panic or divide by zero — it just has nothing to score.
func TestDetectHandlesShortSequences(t *testing.T) {
	cfg := Config{Key: 1, ContextSize: 8, Layers: 2}
	score := Detect(cfg, []int{1, 2, 3})
	if score.Positions != 0 {
		t.Errorf("Positions = %d, want 0 for a sequence shorter than ContextSize", score.Positions)
	}
	if score.Z != 0 {
		t.Errorf("Z = %v, want 0 when nothing was scored", score.Z)
	}
}

// TestDetectAgreesWithGenerateAcrossContinuations checks that Detect is
// reading the same seeds Generate produced mid-sequence, not just at the
// tail: scoring a prefix of a watermarked run should already show a raised
// (if noisier) mean g, not 0.5.
func TestDetectAgreesWithGenerateAcrossContinuations(t *testing.T) {
	gpt := tinyGPT(t)
	cfg := Config{Key: 123, ContextSize: 3, Layers: 3}
	prompt := []int{1, 2, 3}

	out, err := Generate(gpt, cfg, prompt, GenerateOpts{MaxTokens: 300, Temperature: 1, Seed: 2})
	if err != nil {
		t.Fatal(err)
	}
	full := append(append([]int{}, prompt...), out...)
	score := Detect(cfg, full)
	if score.MeanG <= 0.5 {
		t.Errorf("MeanG = %v over %d positions, want > 0.5", score.MeanG, score.Positions)
	}
}
