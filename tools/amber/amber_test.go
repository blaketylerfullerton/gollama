package amber

import "testing"

// The package makes exactly one promise — a bigger number is never a darker
// colour — and every screen that colours data by calling Of is relying on it.
// It's the kind of promise a plausible-looking change to gamma or to the ramp
// length could quietly break, so it's worth a test rather than a comment.
func TestLevelIsMonotonic(t *testing.T) {
	prev := -1
	for i := 0; i <= 1000; i++ {
		s := float64(i) / 1000
		got := LevelOf(s)
		if got < prev {
			t.Fatalf("strength %.3f mapped to level %d, below the %d before it", s, got, prev)
		}
		prev = got
	}
}

func TestLevelSpansTheRamp(t *testing.T) {
	if got := LevelOf(0); got != 0 {
		t.Errorf("LevelOf(0) = %d, want 0 — nothing should still look like nothing", got)
	}
	if got, want := LevelOf(1), len(Ramp)-1; got != want {
		t.Errorf("LevelOf(1) = %d, want %d — certainty should reach the top", got, want)
	}
}

// Attention rows sum to one over every position, so in a sequence of any length
// the weights that carry the structure are small. If the low end of the range
// collapses onto one or two levels the matrix renders as a flat wash, which is
// the failure the gamma exists to prevent.
func TestSmallWeightsStaySeparable(t *testing.T) {
	weights := []float64{0.01, 0.03, 0.08, 0.15, 0.3}
	seen := map[int]bool{}
	for _, w := range weights {
		seen[LevelOf(w)] = true
	}
	if len(seen) < 4 {
		t.Errorf("%v collapsed onto %d levels; the low end needs at least 4", weights, len(seen))
	}
}

func TestAtClamps(t *testing.T) {
	if At(-5) != Ramp[0] {
		t.Error("At below the ramp should return the bottom")
	}
	if At(99) != Ramp[len(Ramp)-1] {
		t.Error("At above the ramp should return the top")
	}
}
