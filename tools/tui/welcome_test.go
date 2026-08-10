package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// The art is laid beside a panel, so a single short line would leave a notch in
// the layout that only shows up at runtime. Every frame has to be the same size
// as every other one too, or the panel beside it would jump on each tick.
func TestLlamaFramesAreEqualSize(t *testing.T) {
	for f, frame := range llamaFrames {
		if len(frame) != llamaHeight {
			t.Errorf("frame %d is %d rows, want %d", f, len(frame), llamaHeight)
		}
		for i, line := range frame {
			if w := lipgloss.Width(line); w != llamaWidth {
				t.Errorf("frame %d line %d is %d cells wide, want %d: %q", f, i, w, llamaWidth, line)
			}
		}
	}
}

// Every tick has to land on a frame that exists, whatever the counter has got up
// to — the nod and the blink index into the same slice from different clocks.
func TestLlamaFrameAtCoversEveryTick(t *testing.T) {
	seen := map[string]bool{}
	for tick := range nodTicks * blinkPeriod * 2 {
		seen[strings.Join(llamaFrameAt(tick), "")] = true
	}
	if len(seen) != len(llamaFrames) {
		t.Errorf("the animation only ever shows %d of the %d frames", len(seen), len(llamaFrames))
	}
}

// A glyph whose rows disagree about their width shifts every letter after it
// along by the difference, which turns the wordmark into a staircase.
func TestWordmarkGlyphsAreRectangular(t *testing.T) {
	for r, g := range glyphs {
		want := lipgloss.Width(g[0])
		for i, row := range g {
			if got := lipgloss.Width(row); got != want {
				t.Errorf("%q row %d is %d cells, want %d: %q", r, i, got, want, row)
			}
		}
	}

	art, width := wordmark("GoLlama")
	lines := strings.Split(art, "\n")
	if len(lines) != wordmarkRows {
		t.Errorf("wordmark is %d rows, want %d", len(lines), wordmarkRows)
	}
	for i, line := range lines {
		if got := lipgloss.Width(line); got != width {
			t.Errorf("row %d is %d cells, want the reported %d", i, got, width)
		}
	}
}

func TestScanCheckpointMissing(t *testing.T) {
	c := ScanCheckpoint(filepath.Join(t.TempDir(), "nope"))
	if c.Present {
		t.Fatal("reported weights in a directory that does not exist")
	}
	if c.Bytes != 0 {
		t.Errorf("Bytes = %d, want 0", c.Bytes)
	}
}

func TestScanCheckpointPresent(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "model.safetensors"), 2048)
	write(t, filepath.Join(dir, "config.json"), 52)

	// The HuggingFace downloader leaves hardlinked copies under .cache; counting
	// them would roughly double the size we report.
	cache := filepath.Join(dir, ".cache")
	if err := os.MkdirAll(cache, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(cache, "blob"), 2048)

	c := ScanCheckpoint(dir)
	if !c.Present {
		t.Fatal("did not find model.safetensors")
	}
	if c.Bytes != 2100 {
		t.Errorf("Bytes = %d, want 2100 (top-level files only)", c.Bytes)
	}
}

// enter is the only way through to the model, and q the only way out; a typo in
// either leaves the user stuck on the splash.
func TestWelcomeKeys(t *testing.T) {
	for _, tc := range []struct {
		key  string
		want Choice
	}{
		{"enter", Run},
		{" ", Run},
		{"q", Quit},
		{"esc", Quit},
		{"ctrl+c", Quit},
	} {
		w := NewWelcome(t.TempDir())
		_, cmd := w.Update(key(tc.key))
		if cmd == nil {
			t.Errorf("%q did not quit the program", tc.key)
		}
		if w.Choice() != tc.want {
			t.Errorf("%q gave choice %v, want %v", tc.key, w.Choice(), tc.want)
		}
	}
}

// An unrelated key must not fall through to a choice — the screen should just
// sit there.
func TestWelcomeIgnoresOtherKeys(t *testing.T) {
	w := NewWelcome(t.TempDir())
	if _, cmd := w.Update(key("x")); cmd != nil {
		t.Error("x quit the program")
	}
	if w.Choice() != Quit {
		t.Errorf("choice = %v before any decision, want the safe default", w.Choice())
	}
}

// A narrow terminal must not produce lines wider than it, or every row wraps and
// the layout falls apart.
func TestViewFitsNarrowTerminal(t *testing.T) {
	// A real, short checkpoint path: t.TempDir() hands back a path long enough to
	// widen the panel past any terminal, which would be testing the wrong thing.
	for _, width := range []int{60, 80, 120} {
		w := NewWelcome("checkpoints/qwen3-0.6b")
		w.Update(tea.WindowSizeMsg{Width: width, Height: 40})
		for _, line := range strings.Split(w.View(), "\n") {
			if got := lipgloss.Width(line); got > width {
				t.Errorf("at width %d a line is %d cells: %q", width, got, line)
			}
		}
	}
}

// Whatever else changes, the screen has to answer "what am I running on".
func TestViewShowsHardware(t *testing.T) {
	w := NewWelcome(t.TempDir())
	view := w.View()
	for _, want := range []string{"This machine", "cores", "memory", "platform", "Weights"} {
		if !strings.Contains(view, want) {
			t.Errorf("view is missing %q", want)
		}
	}
}

func key(s string) tea.KeyMsg {
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	case "ctrl+c":
		return tea.KeyMsg{Type: tea.KeyCtrlC}
	case " ":
		return tea.KeyMsg{Type: tea.KeySpace, Runes: []rune{' '}}
	default:
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
}

func write(t *testing.T, path string, n int) {
	t.Helper()
	if err := os.WriteFile(path, make([]byte, n), 0o644); err != nil {
		t.Fatal(err)
	}
}
