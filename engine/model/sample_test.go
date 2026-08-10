package model

import (
	"math"
	"testing"
)

// logitsFor builds logits whose softmax at temperature 1 is exactly probs.
// log(p) works because softmax is shift-invariant, so no normalization needed.
func logitsFor(probs ...float64) []float32 {
	out := make([]float32, len(probs))
	for i, p := range probs {
		out[i] = float32(math.Log(p))
	}
	return out
}

func TestArgmax(t *testing.T) {
	if got := Argmax([]float32{0.1, 0.9, 0.4}); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
	// Ties go to the first occurrence.
	if got := Argmax([]float32{0.5, 0.5}); got != 0 {
		t.Errorf("got %d, want 0 for a tie", got)
	}
}

func TestGreedyIgnoresSeed(t *testing.T) {
	logits := logitsFor(0.2, 0.5, 0.3)
	for _, seed := range []uint64{1, 2, 99} {
		s := NewSampler(SampleOpts{Temperature: 0, Seed: seed})
		if got := s.Sample(logits); got != 1 {
			t.Errorf("seed %d: got %d, want the argmax 1", seed, got)
		}
	}
}

func TestSameSeedIsReproducible(t *testing.T) {
	logits := logitsFor(0.4, 0.3, 0.2, 0.1)

	draw := func(seed uint64) []int {
		s := NewSampler(SampleOpts{Temperature: 1, Seed: seed})
		out := make([]int, 20)
		for i := range out {
			out[i] = s.Sample(logits)
		}
		return out
	}

	a, b := draw(7), draw(7)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed diverged at draw %d: %d vs %d", i, a[i], b[i])
		}
	}

	// A different seed should not produce an identical 20-token sequence.
	c := draw(8)
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds produced identical sequences — the seed isn't being used")
	}
}

func TestTopKOfOneIsGreedy(t *testing.T) {
	logits := logitsFor(0.1, 0.15, 0.7, 0.05)
	s := NewSampler(SampleOpts{Temperature: 1, TopK: 1, Seed: 3})
	for i := 0; i < 50; i++ {
		if got := s.Sample(logits); got != 2 {
			t.Fatalf("draw %d: got %d, want 2 — top-k=1 leaves no choice", i, got)
		}
	}
}

func TestTopPTruncatesTheTail(t *testing.T) {
	// Cumulative: 0.7, then 0.9. With p=0.75 the second token crosses the
	// threshold and is kept; ids 2 and 3 must never be drawn.
	logits := logitsFor(0.7, 0.2, 0.07, 0.03)
	s := NewSampler(SampleOpts{Temperature: 1, TopP: 0.75, Seed: 11})
	for i := 0; i < 500; i++ {
		if got := s.Sample(logits); got > 1 {
			t.Fatalf("draw %d: got %d, which top-p should have excluded", i, got)
		}
	}
}

func TestApplyTopPKeepsAtLeastOne(t *testing.T) {
	// A p smaller than the most likely token must still leave something to draw.
	cands := []Candidate{{0, 0.9}, {1, 0.1}}
	if got := applyTopP(cands, 0.5); len(got) != 1 || got[0].ID != 0 {
		t.Errorf("got %v, want just the top candidate", got)
	}
}

func TestFiltersDisabledByDefault(t *testing.T) {
	cands := []Candidate{{0, 0.5}, {1, 0.3}, {2, 0.2}}
	if got := applyTopK(cands, 0); len(got) != 3 {
		t.Errorf("TopK=0 should disable the filter, got %d candidates", len(got))
	}
	if got := applyTopP(cands, 0); len(got) != 3 {
		t.Errorf("TopP=0 should disable the filter, got %d candidates", len(got))
	}
	if got := applyTopP(cands, 1); len(got) != 3 {
		t.Errorf("TopP=1 should disable the filter, got %d candidates", len(got))
	}
}

// The sampler should actually respect the distribution, not just stay in range.
func TestSampleMatchesDistribution(t *testing.T) {
	want := []float64{0.6, 0.3, 0.1}
	logits := logitsFor(want...)

	s := NewSampler(SampleOpts{Temperature: 1, Seed: 42})
	const draws = 20000
	counts := make([]int, len(want))
	for i := 0; i < draws; i++ {
		counts[s.Sample(logits)]++
	}

	for i, w := range want {
		got := float64(counts[i]) / draws
		if math.Abs(got-w) > 0.02 {
			t.Errorf("token %d: sampled %.3f of the time, want %.3f", i, got, w)
		}
	}
}

func TestTemperatureChangesSpreadNotOrder(t *testing.T) {
	logits := logitsFor(0.5, 0.3, 0.15, 0.05)

	cold := TopCandidates(logits, 0.5, 4)
	hot := TopCandidates(logits, 2.0, 4)

	// Ranking is invariant: dividing every logit by the same number cannot
	// reorder them.
	for i := range cold {
		if cold[i].ID != hot[i].ID {
			t.Errorf("rank %d: cold picked %d, hot picked %d — temperature must not reorder",
				i, cold[i].ID, hot[i].ID)
		}
	}
	// Low temperature concentrates mass on the leader, high temperature spreads it.
	if cold[0].Prob <= hot[0].Prob {
		t.Errorf("top probability was %.4f cold and %.4f hot, expected cold to be higher",
			cold[0].Prob, hot[0].Prob)
	}
}

func TestTopCandidatesLimitsAndSorts(t *testing.T) {
	logits := logitsFor(0.1, 0.5, 0.25, 0.15)
	got := TopCandidates(logits, 1, 2)
	if len(got) != 2 {
		t.Fatalf("got %d candidates, want 2", len(got))
	}
	if got[0].ID != 1 || got[1].ID != 2 {
		t.Errorf("got ids %d,%d, want 1,2 in descending probability", got[0].ID, got[1].ID)
	}
	var sum float64
	for _, c := range TopCandidates(logits, 1, 4) {
		sum += c.Prob
	}
	if math.Abs(sum-1) > 1e-9 {
		t.Errorf("full distribution sums to %v, want 1", sum)
	}
}
