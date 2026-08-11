package amber

import (
	"math"
	"strconv"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

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

// contrast is the WCAG ratio of a colour against a black terminal, which is the
// worst case the palette has to survive.
func contrast(t *testing.T, c lipgloss.Color) float64 {
	t.Helper()
	hex := string(c)
	if len(hex) != 7 || hex[0] != '#' {
		t.Fatalf("colour %q is not a #rrggbb literal", hex)
	}
	chan_ := func(i int) float64 {
		v, err := strconv.ParseUint(hex[i:i+2], 16, 8)
		if err != nil {
			t.Fatalf("colour %q: %v", hex, err)
		}
		s := float64(v) / 255
		if s <= 0.04045 {
			return s / 12.92
		}
		return math.Pow((s+0.055)/1.055, 2.4)
	}
	l := 0.2126*chan_(1) + 0.7152*chan_(3) + 0.0722*chan_(5)
	return (l + 0.05) / 0.05
}

// The furniture track's second promise, after "grey doesn't claim a magnitude",
// is that anything named for text is actually legible. The levels below the
// floor exist precisely because they aren't, so the split only stays honest if
// something checks which side of it each name is on — this is the kind of thing
// a plausible tweak to one hex value breaks silently, and silently is the whole
// problem, since low-contrast grey looks fine to whoever picked it.
func TestTextLevelsClearTheContrastFloor(t *testing.T) {
	const floor = 4.5
	for _, c := range []struct {
		name  string
		level int
	}{
		{"Muted", Muted},
		{"Body", Body},
		{"Strong", Strong},
	} {
		if got := contrast(t, N(c.level)); got < floor {
			t.Errorf("Neutral %s (%s) is %.2f:1 on black, under the %.1f:1 body text needs",
				c.name, N(c.level), got, floor)
		}
	}
	if got := contrast(t, At(Accent)); got < floor {
		t.Errorf("Accent (%s) is %.2f:1 on black, under %.1f:1 — it carries keys and the wordmark",
			At(Accent), got, floor)
	}
	if got := contrast(t, Alert); got < floor {
		t.Errorf("Alert (%s) is %.2f:1 on black, under %.1f:1", Alert, got, floor)
	}
}

// The structural levels have to stay on the other side of the floor. A border
// that creeps up into text contrast isn't a neutral bug — it's the flatness the
// two tracks were introduced to fix, arriving one hex value at a time.
func TestStructureLevelsStayBelowText(t *testing.T) {
	for _, c := range []struct {
		name  string
		level int
	}{
		{"Edge", Edge},
		{"Rule", Rule},
	} {
		if got := contrast(t, N(c.level)); got >= 4.5 {
			t.Errorf("Neutral %s (%s) is %.2f:1 — bright enough to read, so it will compete with the text",
				c.name, N(c.level), got)
		}
	}
}
