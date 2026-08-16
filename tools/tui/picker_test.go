package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// A 16GB machine with 8GB free: enough headroom that 0.6B fits and 8B doesn't,
// so the verdicts have something to disagree about.
func testSys() sysinfo.Info {
	return sysinfo.Info{
		Host: "test", CPU: "test cpu", Cores: 8,
		MemoryBytes: 16 << 30, AvailableBytes: 8 << 30,
		OS: "darwin", Arch: "arm64", GoVersion: "go1.25.0", GOMAXPROCS: 8,
	}
}

// checkpoint writes just enough of a HuggingFace directory that the picker
// treats it as installed: weights to find, and a config to read the shape from.
func checkpoint(t *testing.T, root, name string, cfg string) string {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(dir, "model.safetensors"), 4096)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

const qwen3Tiny = `{
  "vocab_size": 1000, "hidden_size": 64, "intermediate_size": 128,
  "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
  "head_dim": 32, "max_position_embeddings": 2048, "tie_word_embeddings": true
}`

// The catalog has to list models that aren't downloaded — knowing 4B exists and
// what it would cost is most of the reason to look at this screen — so a fresh
// root must still produce rows.
func TestCatalogListsUninstalledModels(t *testing.T) {
	c := Catalog(t.TempDir())
	if len(c) < len(known)+1 {
		t.Fatalf("catalog has %d entries, want at least %d", len(c), len(known)+1)
	}
	for _, m := range c[:len(known)] {
		if m.Installed {
			t.Errorf("%s reported installed from an empty root", m.Name)
		}
		if m.Repo == "" {
			t.Errorf("%s has no repo, so there is nothing to tell the user to download", m.Name)
		}
	}
	if last := c[len(c)-1]; !last.Demo {
		t.Errorf("last entry is %q, want the built-in random model", last.Name)
	}
}

// Catalog's root has to actually govern where checkpoints are looked for, or
// the paths in the download hints point somewhere the scan never checked.
func TestCatalogFindsInstalledUnderRoot(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-0.6b", qwen3Tiny)

	var got *Model
	catalog := Catalog(root)
	for i := range catalog {
		if catalog[i].Name == "Qwen3-0.6B" {
			got = &catalog[i]
		}
	}
	if got == nil {
		t.Fatal("Qwen3-0.6B is missing from the catalog")
	}
	if !got.Installed {
		t.Fatal("Qwen3-0.6B was not found under the root it was written to")
	}
	if got.Dir != filepath.Join(root, "qwen3-0.6b") {
		t.Errorf("Dir = %q, want it under %q", got.Dir, root)
	}
	// A present checkpoint describes itself; the built-in numbers are only for
	// models with nothing on disk to ask.
	if got.Arch.NLayer != 4 {
		t.Errorf("NLayer = %d, want 4 read from the checkpoint's own config.json", got.Arch.NLayer)
	}
	if got.OnDisk == 0 {
		t.Error("OnDisk is 0 for an installed model")
	}
}

// A checkpoint downloaded by hand still loads, so it still belongs on the list.
func TestCatalogPicksUpStrays(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "something-else", qwen3Tiny)

	var found bool
	for _, m := range Catalog(root) {
		if m.Name == "something-else" {
			found = true
			if !m.Custom || !m.Installed {
				t.Errorf("stray %q: Custom=%v Installed=%v, want both true", m.Name, m.Custom, m.Installed)
			}
		}
	}
	if !found {
		t.Error("a checkpoint in the root that isn't in the catalog was dropped")
	}
}

// A directory with weights but no readable config can't be sized or loaded, so
// listing it would only offer a failure.
func TestCatalogSkipsUnreadableStray(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "broken", "{not json")
	for _, m := range Catalog(root) {
		if m.Name == "broken" {
			t.Error("listed a checkpoint whose config.json does not parse")
		}
	}
}

// The published parameter counts are the whole basis of the memory column. If
// the formula drifts, every number on the right is wrong by the same factor and
// nothing on screen says so.
func TestKnownParameterCounts(t *testing.T) {
	want := map[string]int64{
		"Qwen3-0.6B": 596_000_000,
		"Qwen3-1.7B": 1_720_000_000,
		"Qwen3-4B":   4_020_000_000,
		"Qwen3-8B":   8_190_000_000,
	}
	for _, m := range known {
		w, ok := want[m.Name]
		if !ok {
			t.Fatalf("no expected parameter count for %s", m.Name)
		}
		got := m.Arch.Params()
		// Within 1%: the published figures are themselves rounded.
		if diff := float64(got-w) / float64(w); diff > 0.01 || diff < -0.01 {
			t.Errorf("%s: Params() = %d, want about %d (%.1f%% off)", m.Name, got, w, 100*diff)
		}
	}
}

// The loader widens bf16 to float32 and drops the redundant tied lm head, so
// resident memory is neither the download size nor twice it. Getting this
// backwards is the mistake the screen exists to prevent.
func TestQwen3SizesMatchTheRealCheckpoint(t *testing.T) {
	a := known[0].Arch // Qwen3-0.6B

	// The real model.safetensors is 1,503,300,328 bytes.
	const onDisk = 1_503_300_328
	if diff := float64(a.DiskBytes()-onDisk) / onDisk; diff > 0.01 || diff < -0.01 {
		t.Errorf("DiskBytes() = %d, want about %d (%.1f%% off)", a.DiskBytes(), onDisk, 100*diff)
	}
	if a.ResidentBytes() != a.Params()*4 {
		t.Errorf("ResidentBytes() = %d, want 4 bytes per parameter", a.ResidentBytes())
	}
	// 28 layers x 8 kv heads x 128 dims x 2 tensors x 4 bytes.
	if got := a.KVBytesPerToken(); got != 229_376 {
		t.Errorf("KVBytesPerToken() = %d, want 229376", got)
	}
}

// Opening on a model that can't run means the first enter does nothing, which
// reads as a broken screen.
func TestPickerOpensOnAnInstalledModel(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-1.7b", qwen3Tiny)

	p := NewPicker(root, testSys())
	if got := p.Selection().Name; got != "Qwen3-1.7B" {
		t.Errorf("cursor opened on %q, want the installed model", got)
	}
}

// With nothing downloaded the cursor has nowhere better to be than the top.
func TestPickerOpensAtTopWhenNothingInstalled(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys())
	if p.cursor != 0 {
		t.Errorf("cursor = %d on an empty root, want 0", p.cursor)
	}
}

// Wrapping means one press too many silently teleports the cursor to the other
// end of a list the user was reading top to bottom.
func TestPickerCursorDoesNotWrap(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys())
	for range len(p.models) + 5 {
		p.Update(key("k"))
	}
	if p.cursor != 0 {
		t.Errorf("cursor = %d after running off the top, want 0", p.cursor)
	}
	for range len(p.models) + 5 {
		p.Update(key("j"))
	}
	if want := len(p.models) - 1; p.cursor != want {
		t.Errorf("cursor = %d after running off the bottom, want %d", p.cursor, want)
	}
}

func TestPickerKeys(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-0.6b", qwen3Tiny)

	for _, tc := range []struct {
		key  string
		want Outcome
	}{
		{"enter", Selected},
		{" ", Selected},
		{"b", Back},
		{"esc", Back},
		{"backspace", Back},
		{"q", Cancelled},
		{"ctrl+c", Cancelled},
	} {
		p := NewPicker(root, testSys())
		_, cmd := p.Update(key(tc.key))
		if cmd == nil {
			t.Errorf("%q did not end the screen", tc.key)
		}
		if p.Outcome() != tc.want {
			t.Errorf("%q gave outcome %v, want %v", tc.key, p.Outcome(), tc.want)
		}
	}
}

// Enter on a model that isn't downloaded has to say so. Quitting to a failed
// load, or sitting there doing nothing, are both worse.
// enter on a model with no weights on disk still selects it — Root is the one
// that decides a missing checkpoint means "fetch it first", not this screen.
func TestPickerSelectsUninstalledModel(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys()) // nothing installed
	_, cmd := p.Update(key("enter"))
	if cmd == nil {
		t.Fatal("enter did nothing for a model with no weights on disk")
	}
	if p.Outcome() != Selected {
		t.Fatal("outcome is not Selected for a model whose weights aren't downloaded yet")
	}
}

// The random model needs no weights, so it is always runnable.
func TestPickerAllowsTheDemoModel(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys())
	p.Update(key("G"))
	if !p.Selection().Demo {
		t.Fatalf("end of the list is %q, want the random model", p.Selection().Name)
	}
	if _, cmd := p.Update(key("enter")); cmd == nil || p.Outcome() != Selected {
		t.Error("enter on the built-in model did not select it")
	}
	if dir := p.Selection().Dir; dir != "" {
		t.Errorf("the random model has Dir %q, want empty so main falls back to it", dir)
	}
}

// Everything on the right is derived from the row under the cursor, so moving
// the cursor has to change it. A panel that never updates is worse than none.
func TestMemoryPanelFollowsTheCursor(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys())
	small := p.memory()
	p.Update(key("G")) // the tiny random model
	if p.memory() == small {
		t.Error("the memory panel is identical for a 0.6B model and a 1M one")
	}
}

// The recommendation is the biggest model that still fits comfortably — under
// half the free memory, the same threshold the gauge calls "room to spare". On
// the 8GB-free test machine, 1.7B costs 91% of that and 0.6B costs 39%, so 0.6B
// is the one worth pointing at even though 1.7B is technically bigger.
func TestPickerRecommendsTheBiggestComfortableFit(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys())
	rec := p.recommended()
	if rec < 0 {
		t.Fatal("nothing was recommended on a machine with headroom to spare")
	}
	if got := p.models[rec].Name; got != "Qwen3-0.6B" {
		t.Errorf("recommended %q, want Qwen3-0.6B", got)
	}
}

// The demo model always fits — it's a few kilobytes — so if it were eligible it
// would always be "the" recommendation and the column would never point at a
// real model.
func TestPickerNeverRecommendsTheDemoModel(t *testing.T) {
	p := &Picker{sys: sysinfo.Info{MemoryBytes: 1 << 20, AvailableBytes: 1 << 20}, models: []Model{demoModel}}
	if rec := p.recommended(); rec != -1 {
		t.Errorf("recommended index %d on a list of nothing but the demo model, want -1", rec)
	}
}

// A model past its headroom is a decision, not a suggestion — "too large" has
// to appear somewhere on the row, and the biggest model on a small machine is
// guaranteed to trigger it.
func TestPickerFlagsTooLargeModels(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys()) // 8GB free
	view := p.View()
	if !strings.Contains(view, "too large") {
		t.Error("no model on the list was flagged too large, on a machine that can't fit the 8B")
	}
	if !strings.Contains(view, "recommended") {
		t.Error("nothing was recommended even though smaller models fit comfortably")
	}
}

// A machine that didn't report its memory has nothing to judge fit by, so the
// column has to say so rather than guess.
func TestPickerFitUnknownWithoutMemoryInfo(t *testing.T) {
	p := &Picker{sys: sysinfo.Info{}, models: Catalog(t.TempDir())}
	if rec := p.recommended(); rec != -1 {
		t.Errorf("recommended index %d with no memory info to judge by, want -1", rec)
	}
	if got := p.fit(p.models[0], false); !strings.Contains(got, "—") {
		t.Errorf("fit() = %q, want the unknown marker", got)
	}
}

// The three verdicts are the point of the gauge: "fits", "most of what's free"
// and "won't fit" are three different decisions.
func TestGaugeVerdicts(t *testing.T) {
	sys := testSys() // 8GB free
	for _, tc := range []struct {
		resident int64
		want     string
	}{
		{1 << 30, "room to spare"},
		{5 << 30, "most of what's free"},
		{7<<30 + 512<<20, "tight"},
		{32 << 30, "won't fit"},
	} {
		p := &Picker{sys: sys, models: Catalog(t.TempDir()), w: 100, h: 40}
		if got := p.gauge(tc.resident); !strings.Contains(got, tc.want) {
			t.Errorf("%s resident: gauge says %q, want it to mention %q",
				sysinfo.Bytes(tc.resident), strings.TrimSpace(got), tc.want)
		}
	}
}

// A machine that doesn't report its memory must not produce a bar drawn against
// zero, which would read as "nothing fits".
func TestGaugeWithoutMemoryInfo(t *testing.T) {
	p := &Picker{sys: sysinfo.Info{}, models: Catalog(t.TempDir()), w: 100, h: 40}
	got := p.gauge(1 << 30)
	if strings.Contains(got, "█") || strings.Contains(got, "won't fit") {
		t.Errorf("gauge rendered a verdict with no memory to compare against: %q", got)
	}
	if !strings.Contains(p.headerRow(100), "unknown") {
		t.Error("the header does not say the machine's numbers are unavailable")
	}
}

// The screen has to answer the two questions it was built for, whatever else
// changes around it.
func TestPickerViewShowsTheEssentials(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-0.6b", qwen3Tiny)

	view := NewPicker(root, testSys()).View()
	for _, want := range []string{
		"choose a model to start with",
		"models", "MEMORY AFTER LOAD",
		"test", "resident", "free after", "keep",
		"Qwen3-0.6B",
	} {
		if !strings.Contains(view, want) {
			t.Errorf("view is missing %q", want)
		}
	}
}

// A model that isn't here yet has to come with the command that fetches it,
// or the screen is describing something the user can't act on.
func TestPickerShowsTheDownloadCommand(t *testing.T) {
	p := NewPicker(t.TempDir(), testSys())
	// An explicit size: three panels of this much content need about 34 rows,
	// and below that the description is clipped from the bottom on purpose —
	// see the trimming in View.
	p.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	view := p.View()
	if !strings.Contains(view, "press enter to fetch") {
		t.Error("no mention of fetching the model for one that isn't installed")
	}
	if !strings.Contains(view, "to download") {
		t.Error("the memory panel does not say how big the download is")
	}
}

// Same constraint as the welcome screen: a line wider than the terminal wraps
// and takes the whole layout with it.
func TestPickerFitsNarrowTerminal(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-4b", qwen3Tiny)

	for _, width := range []int{60, 80, 100, 120} {
		p := NewPicker(root, testSys())
		p.Update(tea.WindowSizeMsg{Width: width, Height: 40})
		for row := range len(p.models) {
			for _, line := range strings.Split(p.View(), "\n") {
				if got := lipgloss.Width(line); got > width {
					t.Fatalf("at width %d, row %d has a %d-cell line: %q", width, row, got, line)
				}
			}
			p.Update(key("j"))
		}
	}
}

// A checkpoint directory full of models must not push the panels off screen on
// a terminal that has no room for them all.
func TestPickerScrollsLongList(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"} {
		checkpoint(t, root, name, qwen3Tiny)
	}
	p := NewPicker(root, testSys())
	p.Update(tea.WindowSizeMsg{Width: 100, Height: 28})

	if n := strings.Count(p.list(80), "\n"); n > minVisibleRows+3 {
		t.Errorf("list is %d lines for %d models, want it windowed", n+1, len(p.models))
	}
	p.Update(key("G"))
	if !strings.Contains(p.list(80), p.Selection().Name) {
		t.Error("the cursor scrolled off the visible window")
	}

	// The point of windowing is the height, so a terminal with the room should
	// spend it: all fifteen models at once rather than eight and a footnote.
	p.Update(tea.WindowSizeMsg{Width: 100, Height: 60})
	for _, m := range p.models {
		if !strings.Contains(p.list(80), m.Name) {
			t.Errorf("%q is still scrolled off a 60-row terminal", m.Name)
		}
	}
	if strings.Contains(p.list(80), "more below") {
		t.Error("the list is still windowed on a terminal tall enough for all of it")
	}
}

// Both screens run on the alternate screen, so a view shorter than the terminal
// leaves blank rows below it and one taller than the terminal scrolls the top
// off. It has to be exactly the height it was given.
func TestScreensFillTheTerminal(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-0.6b", qwen3Tiny)

	for _, size := range []tea.WindowSizeMsg{
		{Width: 80, Height: 24},
		{Width: 100, Height: 40},
		{Width: 200, Height: 60},
		{Width: 60, Height: 20},
	} {
		views := map[string]func() string{}

		p := NewPicker(root, testSys())
		p.Update(size)
		views["picker"] = p.View

		w := NewWelcome("checkpoints/qwen3-0.6b")
		w.Update(size)
		views["welcome"] = w.View

		for name, view := range views {
			lines := strings.Split(view(), "\n")
			if len(lines) != size.Height {
				t.Errorf("%s at %dx%d is %d rows, want %d",
					name, size.Width, size.Height, len(lines), size.Height)
			}
			for _, line := range lines {
				if got := lipgloss.Width(line); got > size.Width {
					t.Errorf("%s at %dx%d has a %d-cell line: %q",
						name, size.Width, size.Height, got, line)
				}
			}
		}
	}
}

// Filling the height is only half of it: a panel that stops two thirds of the
// way across a wide terminal looks like the layout gave up.
func TestPickerUsesTheFullWidth(t *testing.T) {
	root := t.TempDir()
	checkpoint(t, root, "qwen3-0.6b", qwen3Tiny)

	p := NewPicker(root, testSys())
	p.Update(tea.WindowSizeMsg{Width: 160, Height: 45})

	var widest int
	for _, line := range strings.Split(p.View(), "\n") {
		widest = max(widest, lipgloss.Width(line))
	}
	// The margin down each side is the only width the layout may leave unused,
	// and the widest line starts after the left one.
	if want := 160 - screenMargin; widest != want {
		t.Errorf("widest line is %d cells on a 160-cell terminal, want %d", widest, want)
	}
}
