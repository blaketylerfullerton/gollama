package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

// fixture builds a small trace by hand. The whole point of the UI reading files
// rather than a live model is that this needs no checkpoint and no inference.
func fixture() *trace.Trace {
	const nLayer = 3
	tr := &trace.Trace{
		Header: trace.Header{
			Version: trace.FormatVersion,
			Model:   "fixture",
			Prompt:  "The capital of France is",
			Tokens: []trace.Token{
				{ID: 785, Text: "The"}, {ID: 6722, Text: " capital"}, {ID: 374, Text: " is"},
			},
			Config: trace.ModelInfo{
				NLayer: nLayer, NEmbed: 8, NHead: 2, NKVHead: 1, HeadDim: 4, VocabSize: 32,
			},
		},
	}

	add := func(e trace.Event) {
		e.Seq = len(tr.Events)
		tr.Events = append(tr.Events, e)
	}

	add(trace.Event{Kind: trace.KindStage, Layer: -1, Name: "token embeddings",
		Tokens: 3, Dims: 8, MeanNorm: 0.81, Preview: []float32{0.1, -0.2, 0.3}})

	for l := 0; l < nLayer; l++ {
		add(trace.Event{Kind: trace.KindStage, Layer: l, Name: "input norm",
			Tokens: 3, Dims: 8, MeanNorm: 6.7, Preview: []float32{1, 2, 3}})
		add(trace.Event{Kind: trace.KindAttention, Layer: l, Head: 0,
			Weights: [][]float32{{1}, {0.8, 0.2}, {0.5, 0.3, 0.2}}})
		add(trace.Event{Kind: trace.KindRotary, Layer: l, Head: 0,
			Before: []float32{3, 4}, After: []float32{4, 3},
			NormIn: 5, NormOut: 5, CosSim: 0.96})
		add(trace.Event{Kind: trace.KindNote, Layer: l, Text: "gate 51% negative"})
		add(trace.Event{Kind: trace.KindStage, Layer: l, Name: "+ mlp residual",
			Tokens: 3, Dims: 8, MeanNorm: float64(10 * (l + 1))})

		// Attribution: head 1 pushes " Paris" hard in the layer where the answer
		// first leads, head 0 pushes against it, and the MLP is nearly inert.
		// A mixture of signs is the case worth having in the fixture — a view
		// that only ever renders positive bars can be wrong about direction and
		// still look right.
		for h := 0; h < 2; h++ {
			sign := float32(1)
			if h == 0 {
				sign = -1
			}
			add(trace.Event{Kind: trace.KindAttribution, Layer: l, Head: h,
				Component: trace.ComponentHead, Norm: 1.5 + float64(h),
				Effects: []trace.Effect{
					{ID: 12095, Text: " Paris", Logit: sign * float32(l+1)},
					{ID: 264, Text: " a", Logit: -sign * 0.5},
				}})
		}
		add(trace.Event{Kind: trace.KindAttribution, Layer: l,
			Component: trace.ComponentMLP, Norm: 0.4,
			Effects: []trace.Effect{
				{ID: 12095, Text: " Paris", Logit: 0.05},
				{ID: 264, Text: " a", Logit: 0.02},
			}})
	}

	add(trace.Event{Kind: trace.KindAttribution, Layer: -1,
		Component: trace.ComponentEmbed, Norm: 0.9,
		Effects: []trace.Effect{
			{ID: 12095, Text: " Paris", Logit: 0.1},
			{ID: 264, Text: " a", Logit: 0.3},
		}})

	// A prediction that changes mid-stack, then matches the final output. The
	// ids have to differ where the text differs, since "first leads" is decided
	// by id rather than by the rendered string.
	preds := []struct {
		id   int
		text string
		prob float64
	}{{264, " a", 0.3}, {12095, " Paris", 0.6}, {12095, " Paris", 0.9}}
	for l, p := range preds {
		add(trace.Event{Kind: trace.KindLogitLens, Layer: l,
			Top:        []trace.Candidate{{ID: p.id, Text: p.text, Prob: p.prob}},
			Entropy:    2.5 - 0.5*float64(l),
			TargetID:   12095,
			TargetText: " Paris",
			TargetRank: 3 - l, // climbing to the front
			TargetProb: p.prob,
		})
	}
	add(trace.Event{Kind: trace.KindLogitLens, Layer: nLayer,
		Top:        []trace.Candidate{{ID: 12095, Text: " Paris", Prob: 0.65}},
		Entropy:    1.1,
		TargetID:   12095,
		TargetText: " Paris",
		TargetRank: 1,
		TargetProb: 0.65,
	})

	return tr
}

// Every view must render at a range of sizes without panicking — index errors in
// terminal layout code are easy to write and invisible until someone resizes.
func TestAllViewsRender(t *testing.T) {
	sizes := []tea.WindowSizeMsg{
		{Width: 100, Height: 30},
		{Width: 40, Height: 10}, // narrow and short
		{Width: 200, Height: 60},
		{Width: 20, Height: 5}, // absurdly small
	}

	for _, size := range sizes {
		for v := view(0); v < numViews; v++ {
			a := newApp(fixture())
			a.Update(size)
			a.view = v
			out := a.View()
			if out == "" {
				t.Errorf("view %v at %dx%d rendered nothing", v, size.Width, size.Height)
			}
			if !strings.Contains(out, "GoLlama inspect") {
				t.Errorf("view %v at %dx%d lost its header", v, size.Width, size.Height)
			}
		}
	}
}

func TestLensViewShowsProgression(t *testing.T) {
	a := newApp(fixture())
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	a.view = viewLens
	out := a.View()

	for _, want := range []string{" a", " Paris", "out"} {
		if !strings.Contains(out, want) {
			t.Errorf("lens view is missing %q", want)
		}
	}
	// The layer where the final answer first takes the lead is the single most
	// informative thing in the view.
	if !strings.Contains(out, "first leads at layer 1") {
		t.Errorf("lens view should call out where the answer first wins:\n%s", out)
	}
}

func TestAttentionViewShowsMaskAndSink(t *testing.T) {
	a := newApp(fixture())
	a.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	a.view = viewAttention
	out := a.View()

	if !strings.Contains(out, "·") {
		t.Error("attention view should mark masked positions")
	}
	if !strings.Contains(out, "_capital") {
		t.Error("attention view should label axes with token text, spaces made visible")
	}
	// Rows 1 and 2 give token 0 weights 0.8 and 0.5, averaging 65%.
	if !strings.Contains(out, "65%") {
		t.Errorf("attention view should report the attention sink share:\n%s", out)
	}
}

func TestStagesViewShowsNormsAndNotes(t *testing.T) {
	a := newApp(fixture())
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	a.view = viewStages
	a.layer = 1
	out := a.View()

	for _, want := range []string{"input norm", "+ mlp residual", "gate 51% negative", "rotary"} {
		if !strings.Contains(out, want) {
			t.Errorf("stages view is missing %q", want)
		}
	}
}

// --- navigation -------------------------------------------------------------

func key(s string) tea.KeyMsg {
	if len(s) == 1 {
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
	switch s {
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
	}
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
}

func TestLayerNavigationClamps(t *testing.T) {
	a := newApp(fixture())
	last := a.config().NLayer - 1

	// Starts on the last layer, where the prediction has settled.
	if a.layer != last {
		t.Errorf("initial layer is %d, want %d", a.layer, last)
	}

	for i := 0; i < 20; i++ {
		a.Update(key("down"))
	}
	if a.layer != last {
		t.Errorf("layer ran past the end: %d, want %d", a.layer, last)
	}

	for i := 0; i < 20; i++ {
		a.Update(key("up"))
	}
	if a.layer != 0 {
		t.Errorf("layer ran below zero: %d", a.layer)
	}
}

func TestHeadNavigationClamps(t *testing.T) {
	a := newApp(fixture())
	for i := 0; i < 20; i++ {
		a.Update(key("right"))
	}
	if want := a.config().NHead - 1; a.head != want {
		t.Errorf("head is %d, want %d", a.head, want)
	}
	for i := 0; i < 20; i++ {
		a.Update(key("left"))
	}
	if a.head != 0 {
		t.Errorf("head is %d, want 0", a.head)
	}
}

func TestViewCycling(t *testing.T) {
	a := newApp(fixture())
	seen := map[view]bool{a.view: true}
	for i := 0; i < int(numViews); i++ {
		a.Update(key("tab"))
		seen[a.view] = true
	}
	if len(seen) != int(numViews) {
		t.Errorf("cycling visited %d of %d views", len(seen), numViews)
	}
	// A full cycle must return to the start.
	if a.view != viewLens {
		t.Errorf("after a full cycle the view is %v, want %v", a.view, viewLens)
	}
}

func TestQuitKeys(t *testing.T) {
	for _, k := range []string{"q", "ctrl+c", "esc"} {
		a := newApp(fixture())
		var msg tea.KeyMsg
		switch k {
		case "ctrl+c":
			msg = tea.KeyMsg{Type: tea.KeyCtrlC}
		case "esc":
			msg = tea.KeyMsg{Type: tea.KeyEsc}
		default:
			msg = key(k)
		}
		if _, cmd := a.Update(msg); cmd == nil {
			t.Errorf("%q should quit", k)
		}
	}
}

// A trace with no logit-lens events should explain itself rather than render an
// empty pane.
func TestLensViewHandlesMissingLens(t *testing.T) {
	tr := fixture()
	var kept []trace.Event
	for _, e := range tr.Events {
		if e.Kind != trace.KindLogitLens {
			kept = append(kept, e)
		}
	}
	tr.Events = kept

	a := newApp(tr)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.view = viewLens
	if out := a.View(); !strings.Contains(out, "No logit-lens events") {
		t.Errorf("expected an explanation, got:\n%s", out)
	}
}

func TestAttentionViewHandlesMissingHead(t *testing.T) {
	a := newApp(fixture())
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.view = viewAttention
	a.head = 1 // fixture only records head 0
	if out := a.View(); !strings.Contains(out, "No attention weights") {
		t.Errorf("expected an explanation, got:\n%s", out)
	}
}

func TestWindowKeepsSelectionVisible(t *testing.T) {
	for _, tc := range []struct{ total, sel, height int }{
		{29, 0, 10}, {29, 28, 10}, {29, 14, 10}, {5, 2, 10}, {1, 0, 1},
	} {
		lo, hi := window(tc.total, tc.sel, tc.height)
		if lo < 0 || hi > tc.total || lo > hi {
			t.Errorf("window(%d,%d,%d) = [%d,%d): out of range", tc.total, tc.sel, tc.height, lo, hi)
		}
		if tc.sel < tc.total && (tc.sel < lo || tc.sel >= hi) {
			t.Errorf("window(%d,%d,%d) = [%d,%d): selection not visible",
				tc.total, tc.sel, tc.height, lo, hi)
		}
	}
}

// The attribution view has to show direction, not just magnitude: a head that
// pushes the answer down is as much of a finding as one that pushes it up, and
// the two are indistinguishable if only the size is drawn.
func TestAttributionViewShowsBothDirections(t *testing.T) {
	a := newApp(fixture())
	a.Update(tea.WindowSizeMsg{Width: 110, Height: 40})
	a.view = viewAttribution
	a.layer = 1
	out := a.View()

	for _, want := range []string{"head 0", "head 1", "mlp", " Paris"} {
		if !strings.Contains(out, want) {
			t.Errorf("attribution view is missing %q:\n%s", want, out)
		}
	}
	// Head 1 is +2 in layer 1, head 0 is -2.
	if !strings.Contains(out, "+2.000") || !strings.Contains(out, "-2.000") {
		t.Errorf("attribution view should show signed effects:\n%s", out)
	}
	if !strings.Contains(out, "largest across the whole pass") {
		t.Errorf("attribution view should rank contributors across layers:\n%s", out)
	}
}

// Attribution is opt-in, so a trace made without it must say so rather than
// render an empty pane.
func TestAttributionViewHandlesMissingAttribution(t *testing.T) {
	tr := fixture()
	var kept []trace.Event
	for _, e := range tr.Events {
		if e.Kind != trace.KindAttribution {
			kept = append(kept, e)
		}
	}
	tr.Events = kept

	a := newApp(tr)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.view = viewAttribution
	if out := a.View(); !strings.Contains(out, "No attribution") {
		t.Errorf("expected an explanation, got:\n%s", out)
	}
}

// The lens view gained two columns whose whole point is that they disagree with
// the top row: an answer can be climbing the ranks well before it leads.
func TestLensViewShowsRankAndEntropy(t *testing.T) {
	a := newApp(fixture())
	a.Update(tea.WindowSizeMsg{Width: 110, Height: 40})
	a.view = viewLens
	out := a.View()

	if !strings.Contains(out, "#3") {
		t.Errorf("lens view should show the target's rank at layers where it isn't leading:\n%s", out)
	}
	if !strings.Contains(out, "2.50") {
		t.Errorf("lens view should show entropy:\n%s", out)
	}
}
