package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

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

// Every real Qwen3 checkpoint above 0.6B ships sharded — model.safetensors
// never exists, only model-0000N-of-0000M.safetensors plus an index — so a
// scan that only looked for the single-file name reported a fully and
// correctly downloaded checkpoint as not present at all.
func TestScanCheckpointPresentSharded(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), 1024)
	write(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), 1024)
	write(t, filepath.Join(dir, "model.safetensors.index.json"), 52)
	write(t, filepath.Join(dir, "config.json"), 52)

	c := ScanCheckpoint(dir)
	if !c.Present {
		t.Fatal("did not find a sharded checkpoint with no single model.safetensors")
	}
	if c.Bytes != 2152 {
		t.Errorf("Bytes = %d, want 2152 (both shards, the index, and config.json)", c.Bytes)
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

// esc means "back" on every other screen. Wiring it to quit here too — the
// screen every one of those backs out to — would mean the reflex trained
// everywhere else silently closes the program the first time it's used out
// of habit on the one screen that has nowhere left to back out to.
func TestWelcomeEscDoesNotQuit(t *testing.T) {
	w := NewWelcome(t.TempDir())
	if _, cmd := w.Update(key("esc")); cmd != nil {
		t.Error("esc quit the program")
	}
}

// The menu opens on Chat — the top row — and pressing enter without
// touching an arrow key should lead straight to it.
func TestWelcomeOpensOnChat(t *testing.T) {
	w := NewWelcome(t.TempDir())
	if w.cursor != 0 {
		t.Errorf("cursor = %d, want 0 (Chat)", w.cursor)
	}
	_, cmd := w.Update(key("enter"))
	if cmd == nil {
		t.Fatal("enter did not end the screen")
	}
	if w.Choice() != Run {
		t.Errorf("Choice() = %v, want Run", w.Choice())
	}
	if w.Tool() != ToolChat {
		t.Errorf("Tool() = %v, want ToolChat", w.Tool())
	}
}

// Every row but Machine picks a tool, and enter always means Run for those —
// Tool() is what tells them apart, not Choice(). Machine is highlightable
// but leads nowhere: it exists for its detail panel, not to be run.
func TestWelcomeRowsPickATool(t *testing.T) {
	want := []Tool{ToolChat, ToolAttention, ToolLens, ToolAttribution, ToolAblation, ToolWatermark, ToolModel}
	if len(want)+1 != len(menuItems) {
		t.Fatalf("test covers %d runnable rows + Machine, menu has %d rows", len(want), len(menuItems))
	}
	for i, tool := range want {
		w := NewWelcome(t.TempDir())
		for range i {
			w.Update(key("j"))
		}
		if w.cursor != i {
			t.Fatalf("cursor = %d after %d downs, want %d", w.cursor, i, i)
		}
		_, cmd := w.Update(key("enter"))
		if cmd == nil {
			t.Fatalf("row %d: enter did not end the screen", i)
		}
		if w.Choice() != Run {
			t.Errorf("row %d: Choice() = %v, want Run", i, w.Choice())
		}
		if w.Tool() != tool {
			t.Errorf("row %d: Tool() = %v, want %v", i, w.Tool(), tool)
		}
	}
}

// Machine is the one row that leads nowhere — enter on it should not end the
// screen, since there's no tool behind it to run.
func TestWelcomeMachineRowDoesNotRun(t *testing.T) {
	w := NewWelcome(t.TempDir())
	for range len(menuItems) - 1 {
		w.Update(key("j"))
	}
	if got, want := menuItems[w.cursor].title, "Machine"; got != want {
		t.Fatalf("cursor landed on %q, want %q", got, want)
	}
	if _, cmd := w.Update(key("enter")); cmd != nil {
		t.Error("enter on Machine ended the screen")
	}
	if w.Choice() != Quit {
		t.Errorf("Choice() = %v after enter on Machine, want the safe default", w.Choice())
	}
}

// About and history are footer keys now, not cursor rows — pressing them
// works the same regardless of which tool is highlighted.
func TestWelcomeAboutKeyIgnoresCursor(t *testing.T) {
	w := NewWelcome(t.TempDir())
	w.Update(key("j"))
	w.Update(key("j"))
	_, cmd := w.Update(key("a"))
	if cmd == nil {
		t.Fatal("a did not end the screen")
	}
	if w.Choice() != ShowAbout {
		t.Errorf("Choice() = %v, want ShowAbout", w.Choice())
	}
}

func TestWelcomeHistoryKey(t *testing.T) {
	w := NewWelcome(t.TempDir())
	_, cmd := w.Update(key("h"))
	if cmd == nil {
		t.Fatal("h did not end the screen")
	}
	if w.Choice() != ShowHistory {
		t.Errorf("Choice() = %v, want ShowHistory", w.Choice())
	}
}

// Five rows, no wrapping — same reasoning as the picker's list: one press too
// many past either end should not silently jump to the other end.
func TestWelcomeCursorDoesNotWrap(t *testing.T) {
	w := NewWelcome(t.TempDir())
	w.Update(key("k")) // already at the top
	if w.cursor != 0 {
		t.Errorf("cursor = %d after up from the top, want 0", w.cursor)
	}
	for range len(menuItems) + 3 {
		w.Update(key("j"))
	}
	if want := len(menuItems) - 1; w.cursor != want {
		t.Errorf("cursor = %d after running off the bottom, want %d", w.cursor, want)
	}
}

// The whole point of the menu is that each row explains what pressing enter
// on it leads to — Attention's blurb by default, Logit Lens's once
// highlighted — and every row's title is visible at once, tools and Other
// alike.
//
// Resized taller than NewWelcome's own default: the selector list, its
// detail box, and the bottom machine row is more content than fits a bare
// 32-row terminal, the same reason TestViewFitsNarrowTerminal exercises its
// own explicit sizes rather than relying on the default.
func TestWelcomeShowsBothMenuRows(t *testing.T) {
	w := NewWelcome(t.TempDir())
	w.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	view := w.View()
	for _, want := range []string{"Chat", "Ablation", "Attention", "Other", "Model", "Machine"} {
		if !strings.Contains(view, want) {
			t.Errorf("view is missing %q", want)
		}
	}
	// "turn by turn" rather than a longer phrase from the blurb: word-wrap in
	// the panel can legitimately split a phrase across lines at some widths,
	// and this test cares that the teaser rendered, not exactly how it wrapped.
	if !strings.Contains(view, "turn by turn") {
		t.Error("Chat's teaser is not showing by default")
	}

	w.Update(key("j"))
	view = w.View()
	if !strings.Contains(view, "magnitude") {
		t.Error("highlighting \"Attention\" did not show its teaser")
	}

	w.Update(key("j"))
	view = w.View()
	if !strings.Contains(view, "stopped early") {
		t.Error("highlighting \"Logit Lens\" did not show its teaser")
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

// Whatever else changes, the screen has to answer "what am I running on" —
// the bottom row says so regardless of which menu row is highlighted.
func TestViewShowsHardware(t *testing.T) {
	w := NewWelcome(t.TempDir())
	view := w.View()
	for _, want := range []string{"host", "cores", "memory", "platform", "weights"} {
		if !strings.Contains(view, want) {
			t.Errorf("view is missing %q", want)
		}
	}
}

// key builds the tea.KeyMsg for s, shared by every test in this package —
// including inspect_test.go's, which is why it also knows the arrow keys and
// tab: welcome's own tests never need them, but reaching for a second,
// near-identical helper there just to cover a few more cases wasn't worth it.
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
	case "up":
		return tea.KeyMsg{Type: tea.KeyUp}
	case "down":
		return tea.KeyMsg{Type: tea.KeyDown}
	case "left":
		return tea.KeyMsg{Type: tea.KeyLeft}
	case "right":
		return tea.KeyMsg{Type: tea.KeyRight}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
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
