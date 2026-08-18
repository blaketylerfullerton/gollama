package tui

import (
	"errors"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

// fixture builds a small trace by hand. The whole point of the UI reading
// files rather than a live model is that this needs no checkpoint and no
// inference.
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

// --- rendering ---------------------------------------------------------------

// allInspectViews is every locked view a welcome-menu tool can open Inspect
// on — see toolInitialView. There's no cycling between them any more, so
// tests that want to check every view exercise this list directly rather
// than looping a live cursor through them.
var allInspectViews = []inspectView{viewAblation, viewAttention, viewAttribution, viewLens}

// Every view must render at a range of sizes without panicking — index errors in
// terminal layout code are easy to write and invisible until someone resizes.
func TestInspectAllViewsRender(t *testing.T) {
	sizes := []tea.WindowSizeMsg{
		{Width: 100, Height: 30},
		{Width: 40, Height: 10}, // narrow and short
		{Width: 200, Height: 60},
		{Width: 20, Height: 5}, // absurdly small
	}

	for _, size := range sizes {
		for _, v := range allInspectViews {
			a := NewInspectFile(fixture())
			a.Update(size)
			a.view = v
			out := a.View()
			if out == "" {
				t.Errorf("view %v at %dx%d rendered nothing", v, size.Width, size.Height)
			}
			// The pinned bar has to survive even when the terminal is too
			// short to also fit the header — screen() clips the body rather
			// than push the toolbar (and the quit key on it) off screen; see
			// layout.go.
			if !strings.Contains(out, "quit") {
				t.Errorf("view %v at %dx%d lost its pinned control bar", v, size.Width, size.Height)
			}
			if size.Height >= 10 && !strings.Contains(out, "GoLlama inspect") {
				t.Errorf("view %v at %dx%d lost its header", v, size.Width, size.Height)
			}
		}
	}
}

// The bar of keys along the bottom used to float wherever the view above it
// stopped — it moved a different number of rows up the screen depending on
// how tall the current view's content was. Now that View() goes through the
// same screen() frame as every other screen (see layout.go), the bar always
// lands on the last row and the frame always fills the terminal to exactly
// its height, at every size and in every phase.
//
// This only checks row count, not line width the way picker_test.go's
// TestScreensFillTheTerminal does for the picker and welcome screens: those
// wrap their content in width-bound panels, but Inspect's per-view rows
// (attention bars, attribution tables, and the like) never have been — a
// separate, pre-existing gap this change doesn't touch.
func TestInspectFillsTheTerminal(t *testing.T) {
	sizes := []tea.WindowSizeMsg{
		{Width: 80, Height: 24},
		{Width: 100, Height: 40},
		{Width: 200, Height: 60},
		{Width: 60, Height: 20},
	}

	for _, size := range sizes {
		for _, v := range allInspectViews {
			a := NewInspectFile(fixture())
			a.Update(size)
			a.view = v
			checkFillsTerminal(t, "inspect/"+v.String(), a.View(), size)
		}

		// The loading and error phases go through different early-return
		// branches in View() — both have to fill the frame too.
		ch := make(chan tea.Msg, 4)
		loading := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewLens)
		loading.Update(size)
		checkFillsTerminal(t, "inspect/loading", loading.View(), size)

		erred := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewLens)
		erred.Update(size)
		erred.Update(InspectErr{Err: errors.New("boom")})
		checkFillsTerminal(t, "inspect/error", erred.View(), size)
	}
}

// checkFillsTerminal asserts view is exactly size.Height rows — the frame
// invariant screen() is responsible for, whatever the content inside it does.
func checkFillsTerminal(t *testing.T, name, view string, size tea.WindowSizeMsg) {
	t.Helper()
	lines := strings.Split(view, "\n")
	if len(lines) != size.Height {
		t.Errorf("%s at %dx%d is %d rows, want %d", name, size.Width, size.Height, len(lines), size.Height)
	}
}

func TestInspectLensViewShowsProgression(t *testing.T) {
	a := NewInspectFile(fixture())
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

// The fixture's third token attends 0.5/0.3/0.2 over The/capital/is — enough
// to check the bar labels, the top-attended callout, and the head's sink
// stat all in one query token's row.
func TestInspectAttentionBarsViewShowsWeightsAndSink(t *testing.T) {
	a := NewInspectFile(fixture())
	a.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	a.view = viewAttention
	a.attnQuery = 2
	out := a.View()

	if !strings.Contains(out, "_capital") {
		t.Error("attention view should label bars with token text, spaces made visible")
	}
	if !strings.Contains(out, "50.0%") {
		t.Errorf("attention view should show the query token's own weights:\n%s", out)
	}
	if !strings.Contains(out, `attends most to "The"`) {
		t.Errorf("attention view should call out the top-weighted token:\n%s", out)
	}
	// Rows 1 and 2 give token 0 weights 0.8 and 0.5, averaging 65%.
	if !strings.Contains(out, "65%") {
		t.Errorf("attention view should report the attention sink share:\n%s", out)
	}
}

// tab/shift+tab used to cycle the whole screen's view; now that each tool
// locks its view, they're free to move the attention screen's focused query
// token instead — see attnQueryCount.
func TestInspectAttentionTokenCycling(t *testing.T) {
	a := NewInspectFile(fixture())
	a.view = viewAttention // fixture rows: 3 query tokens at every layer

	if a.attnQuery != 0 {
		t.Fatalf("attnQuery starts at %d, want 0", a.attnQuery)
	}
	a.Update(key("tab"))
	if a.attnQuery != 1 {
		t.Errorf("tab should advance attnQuery, got %d", a.attnQuery)
	}
	a.Update(key("tab"))
	a.Update(key("tab")) // 1 -> 2 -> wraps to 0
	if a.attnQuery != 0 {
		t.Errorf("attnQuery should wrap forward, got %d", a.attnQuery)
	}
	a.Update(key("shift+tab"))
	if a.attnQuery != 2 {
		t.Errorf("shift+tab should wrap backward, got %d", a.attnQuery)
	}
}

// "t" toggles the trace view, which follows one query token down through
// every layer instead of showing one layer/head's bars.
func TestInspectAttentionTraceViewFollowsQueryAcrossLayers(t *testing.T) {
	a := NewInspectFile(fixture())
	a.Update(tea.WindowSizeMsg{Width: 120, Height: 40})
	a.view = viewAttention
	a.attnQuery = 2
	a.Update(key("t"))
	if !a.attnTrace {
		t.Fatal("t should toggle the trace view on")
	}

	out := a.View()
	for _, want := range []string{"tracing", `"_is"`, "The", "50.0%"} {
		if !strings.Contains(out, want) {
			t.Errorf("trace view is missing %q:\n%s", want, out)
		}
	}
	// Every layer in the fixture records identical weights for head 0, so the
	// top target never moves — the view should say so.
	if !strings.Contains(out, "settles on") {
		t.Errorf("trace view should call out where the target stabilizes:\n%s", out)
	}

	a.Update(key("t"))
	if a.attnTrace {
		t.Error("t should toggle the trace view back off")
	}
}

// --- navigation -------------------------------------------------------------

func TestInspectLayerNavigationClamps(t *testing.T) {
	a := NewInspectFile(fixture())
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

func TestInspectHeadNavigationClamps(t *testing.T) {
	a := NewInspectFile(fixture())
	a.view = viewAttention // lens has no head axis, see handleKey
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

// Each tool locks Inspect to its own view for the rest of the session — no
// in-screen switcher left to wander into the others, including the keys that
// used to drive one.
func TestInspectViewIsLockedToTheTool(t *testing.T) {
	for _, v := range allInspectViews {
		a := NewInspectFile(fixture())
		a.view = v
		for _, k := range []string{"tab", "shift+tab", "]", "[", "1", "2", "3", "4", "5"} {
			a.Update(key(k))
			if a.view != v {
				t.Errorf("key %q changed the locked view from %v to %v", k, v, a.view)
			}
		}
	}
}

// esc backs out to the picker, q and ctrl+c quit outright — either way the
// screen has to end, which every other screen in this package signals with a
// non-nil command rather than tea.Quit directly (see done in root.go).
func TestInspectQuitKeys(t *testing.T) {
	for _, k := range []string{"q", "ctrl+c", "esc"} {
		a := NewInspectFile(fixture())
		if _, cmd := a.Update(key(k)); cmd == nil {
			t.Errorf("%q should end the screen", k)
		}
	}
}

// A trace with no logit-lens events should explain itself rather than render an
// empty pane.
func TestInspectLensViewHandlesMissingLens(t *testing.T) {
	tr := fixture()
	var kept []trace.Event
	for _, e := range tr.Events {
		if e.Kind != trace.KindLogitLens {
			kept = append(kept, e)
		}
	}
	tr.Events = kept

	a := NewInspectFile(tr)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.view = viewLens
	if out := a.View(); !strings.Contains(out, "No logit-lens events") {
		t.Errorf("expected an explanation, got:\n%s", out)
	}
}

func TestInspectAttentionViewHandlesMissingHead(t *testing.T) {
	a := NewInspectFile(fixture())
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.view = viewAttention
	a.head = 1 // fixture only records head 0
	if out := a.View(); !strings.Contains(out, "No attention weights") {
		t.Errorf("expected an explanation, got:\n%s", out)
	}
}

func TestInspectWindowKeepsSelectionVisible(t *testing.T) {
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
func TestInspectAttributionViewShowsBothDirections(t *testing.T) {
	a := NewInspectFile(fixture())
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
func TestInspectAttributionViewHandlesMissingAttribution(t *testing.T) {
	tr := fixture()
	var kept []trace.Event
	for _, e := range tr.Events {
		if e.Kind != trace.KindAttribution {
			kept = append(kept, e)
		}
	}
	tr.Events = kept

	a := NewInspectFile(tr)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.view = viewAttribution
	if out := a.View(); !strings.Contains(out, "No attribution") {
		t.Errorf("expected an explanation, got:\n%s", out)
	}
}

// The lens view gained two columns whose whole point is that they disagree with
// the top row: an answer can be climbing the ranks well before it leads.
func TestInspectLensViewShowsRankAndEntropy(t *testing.T) {
	a := NewInspectFile(fixture())
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

// --- live mode ---------------------------------------------------------------

// feed drives a in the way bubbletea would, one message at a time. Any command
// a message returns is dropped: a command that reads the events channel would
// block here, and the point of feeding manually is to skip that.
func feed(t *testing.T, a *Inspect, msgs ...tea.Msg) {
	t.Helper()
	for _, m := range msgs {
		a.Update(m)
	}
}

// runCmd executes cmd the way bubbletea's own runtime does: a tea.Batch
// returns a tea.BatchMsg rather than running its sub-commands itself — that
// unwrapping is normally the runtime's job — so a test that wants a batched
// command's side effects (like requestPreview riding alongside textinput's
// own cmd) has to do it manually.
func runCmd(cmd tea.Cmd) {
	if cmd == nil {
		return
	}
	if batch, ok := cmd().(tea.BatchMsg); ok {
		for _, c := range batch {
			runCmd(c)
		}
	}
}

// Before the first step lands there's nothing to render, but the UI still has to
// show something rather than panic on a nil step.
func TestInspectLiveShowsStatusBeforeFirstStep(t *testing.T) {
	ch := make(chan tea.Msg, 4)
	a := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewLens)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})

	out := a.View()
	if !strings.Contains(out, "starting") {
		t.Errorf("expected initial status, got:\n%s", out)
	}

	a.status = "loading weights (1.5GB)…"
	if out := a.View(); !strings.Contains(out, "loading weights") {
		t.Errorf("expected the status to render, got:\n%s", out)
	}
	// No step yet, so every view must degrade rather than index into nothing.
	for _, v := range allInspectViews {
		a.view = v
		if a.View() == "" {
			t.Errorf("view %v rendered nothing with no steps", v)
		}
	}
}

func TestInspectLiveAccumulatesSteps(t *testing.T) {
	ch := make(chan tea.Msg, 8)
	a := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewLens)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})

	feed(t, a,
		InspectStatus("prefill…"),
		InspectStep{Label: "prefill", Tr: fixture()},
		InspectStep{Label: " Paris", Tr: fixture()},
		InspectStep{Label: ",", Tr: fixture()},
		InspectRunDone{},
	)

	if len(a.steps) != 3 {
		t.Fatalf("collected %d steps, want 3", len(a.steps))
	}
	// The view should follow the newest step as it arrives.
	if a.cur != 2 {
		t.Errorf("cur is %d, want 2 (should track the latest step)", a.cur)
	}
	if a.phase != inspectBrowsing {
		t.Errorf("InspectRunDone should return to browsing, phase is %v", a.phase)
	}
	if a.status != "" {
		t.Errorf("status should be cleared, got %q", a.status)
	}

	// The step strip is what makes multi-token traces navigable.
	out := a.View()
	for _, want := range []string{"steps", "prefill", "Paris"} {
		if !strings.Contains(out, want) {
			t.Errorf("header is missing %q:\n%s", want, out)
		}
	}
}

func TestInspectStepNavigationClamps(t *testing.T) {
	ch := make(chan tea.Msg, 8)
	a := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewLens)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	feed(t, a,
		InspectStep{Label: "prefill", Tr: fixture()},
		InspectStep{Label: " Paris", Tr: fixture()},
	)

	for i := 0; i < 10; i++ {
		a.Update(key("n"))
	}
	if a.cur != 1 {
		t.Errorf("step ran past the end: %d, want 1", a.cur)
	}
	for i := 0; i < 10; i++ {
		a.Update(key("p"))
	}
	if a.cur != 0 {
		t.Errorf("step ran below zero: %d", a.cur)
	}
}

func TestInspectLiveErrorIsShown(t *testing.T) {
	ch := make(chan tea.Msg, 4)
	a := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewLens)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.Update(InspectErr{Err: errors.New("no checkpoint at nowhere/")})

	out := a.View()
	if !strings.Contains(out, "no checkpoint at nowhere/") {
		t.Errorf("expected the error text, got:\n%s", out)
	}
	if !strings.Contains(out, "esc to go back") {
		t.Error("an error screen should say how to get out of it")
	}
	if a.phase == inspectRunning {
		t.Error("an error should stop the running state")
	}
}

// Navigation must not run off the end when a later step has fewer layers than
// the one currently selected — decode steps are traced separately and a
// truncated trace is possible.
func TestInspectNavigationSurvivesShorterStep(t *testing.T) {
	small := &trace.Trace{Header: trace.Header{
		Config: trace.ModelInfo{NLayer: 1, NHead: 1},
	}}
	ch := make(chan tea.Msg, 4)
	a := NewInspectLive(ch, make(chan InspectRequest, 1), "", 3, viewAttention)
	a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})

	feed(t, a, InspectStep{Label: "big", Tr: fixture()})
	a.layer = 2
	a.head = 1

	feed(t, a, InspectStep{Label: "small", Tr: small})
	// Arrow keys clamp against the *current* step's dimensions.
	a.Update(key("down"))
	a.Update(key("right"))

	if a.layer > 0 {
		t.Errorf("layer is %d but the active step has 1 layer", a.layer)
	}
	if a.head > 0 {
		t.Errorf("head is %d but the active step has 1 head", a.head)
	}
	if a.View() == "" {
		t.Error("rendering a shorter step produced nothing")
	}
}

// --- prompt editing -----------------------------------------------------------

func inspectLiveApp(prompt string, n int) (*Inspect, chan InspectRequest) {
	reqs := make(chan InspectRequest, 4)
	a := NewInspectLive(make(chan tea.Msg, 8), reqs, prompt, n, viewLens)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	return a, reqs
}

func TestInspectStartsInLoadingPhase(t *testing.T) {
	a, _ := inspectLiveApp("hi", 3)
	if a.phase != inspectLoading {
		t.Errorf("phase is %v, want inspectLoading", a.phase)
	}
	// Keystrokes must not reach the prompt field before the model is ready.
	a.Update(key("x"))
	if got := a.input.Value(); got != "hi" {
		t.Errorf("input changed to %q while still loading", got)
	}
}

// InspectReady is what flips the UI from "loading" to accepting input.
func TestInspectReadyEnablesEditing(t *testing.T) {
	a, _ := inspectLiveApp("hi", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})

	if a.phase != inspectEditing {
		t.Fatalf("phase is %v, want inspectEditing", a.phase)
	}
	if !a.input.Focused() {
		t.Error("the prompt field should take focus when the model is ready")
	}
	if out := a.View(); !strings.Contains(out, "enter to run") {
		t.Errorf("editing view should say how to run:\n%s", out)
	}
}

func TestInspectTypingEditsThePrompt(t *testing.T) {
	a, reqs := inspectLiveApp("", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})

	// Every keystroke that changes the prompt asks the engine to tokenize it,
	// so the live preview stays in sync — run each returned command the way
	// bubbletea would, and drain the channel as they land so a run of
	// keystrokes never blocks on a small buffer.
	previews := 0
	drain := func() {
		for {
			select {
			case req := <-reqs:
				if !req.Preview {
					t.Errorf("expected a preview request while editing, got %+v", req)
				}
				previews++
			default:
				return
			}
		}
	}
	press := func(k string) {
		_, cmd := a.Update(key(k))
		runCmd(cmd)
		drain()
	}

	for _, r := range "cats" {
		press(string(r))
	}
	if got := a.input.Value(); got != "cats" {
		t.Errorf("input is %q, want %q", got, "cats")
	}

	// Navigation keys are text while editing, not commands.
	before := a.layer
	press("j")
	if a.layer != before {
		t.Error("j should type a character while editing, not move the layer")
	}
	if got := a.input.Value(); got != "catsj" {
		t.Errorf("input is %q, want %q", got, "catsj")
	}

	if previews == 0 {
		t.Error("no preview request was sent for any keystroke")
	}
}

func TestInspectEnterSubmitsThePrompt(t *testing.T) {
	a, reqs := inspectLiveApp("the sky is", 4)
	a.Update(InspectReady{Info: fixture().Header.Config})

	_, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("enter should produce a command that sends the request")
	}
	cmd() // bubbletea would run this on its own goroutine

	select {
	case req := <-reqs:
		if req.Prompt != "the sky is" {
			t.Errorf("submitted %q", req.Prompt)
		}
		if req.MaxTokens != 4 {
			t.Errorf("maxTokens is %d, want 4", req.MaxTokens)
		}
	default:
		t.Fatal("nothing was sent to the engine")
	}

	if a.phase != inspectRunning {
		t.Errorf("phase is %v, want inspectRunning", a.phase)
	}
	if a.input.Focused() {
		t.Error("the field should blur while inference runs")
	}
}

// Submitting again has to discard the previous run, or the step strip would
// mix two prompts together.
func TestInspectSubmitClearsPreviousRun(t *testing.T) {
	a, reqs := inspectLiveApp("first", 2)
	a.Update(InspectReady{Info: fixture().Header.Config})
	feed(t, a,
		InspectStep{Label: "prefill", Tr: fixture()},
		InspectStep{Label: " x", Tr: fixture()},
		InspectRunDone{},
	)
	if len(a.steps) != 2 {
		t.Fatalf("setup: got %d steps", len(a.steps))
	}

	a.Update(key("i")) // back to the prompt
	a.input.SetValue("second")
	_, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	cmd()
	<-reqs

	if len(a.steps) != 0 {
		t.Errorf("previous run's %d steps survived the resubmit", len(a.steps))
	}
	if a.cur != 0 {
		t.Errorf("cur is %d, want 0", a.cur)
	}
}

func TestInspectEmptyPromptIsNotSubmitted(t *testing.T) {
	a, reqs := inspectLiveApp("", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})

	_, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil {
		cmd()
	}
	select {
	case req := <-reqs:
		t.Errorf("an empty prompt was submitted: %+v", req)
	default:
	}
	if a.phase != inspectEditing {
		t.Errorf("phase is %v, should still be editing", a.phase)
	}
}

func TestInspectIReturnsToThePrompt(t *testing.T) {
	a, _ := inspectLiveApp("hi", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})
	feed(t, a, InspectStep{Label: "prefill", Tr: fixture()}, InspectRunDone{})

	if a.phase != inspectBrowsing {
		t.Fatalf("phase is %v, want inspectBrowsing", a.phase)
	}
	a.Update(key("i"))
	if a.phase != inspectEditing {
		t.Errorf("i should return to the prompt, phase is %v", a.phase)
	}
	if !a.input.Focused() {
		t.Error("the field should regain focus")
	}
}

// esc from the prompt goes to browsing, but only when there's something to
// browse — otherwise it would leave the user on a dead screen.
func TestInspectEscFromPromptNeedsResults(t *testing.T) {
	a, _ := inspectLiveApp("hi", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})

	a.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if a.phase != inspectEditing {
		t.Error("esc with no results should keep the user in the prompt")
	}

	feed(t, a, InspectStep{Label: "prefill", Tr: fixture()})
	a.phase = inspectEditing
	a.input.Focus()
	a.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if a.phase != inspectBrowsing {
		t.Errorf("esc with results should switch to browsing, phase is %v", a.phase)
	}
}

func TestInspectTokenCountAdjustment(t *testing.T) {
	a, _ := inspectLiveApp("hi", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})
	a.phase = inspectBrowsing // +/- are browsing-mode keys

	a.Update(key("+"))
	if a.maxTokens != 4 {
		t.Errorf("maxTokens is %d, want 4", a.maxTokens)
	}
	for i := 0; i < 10; i++ {
		a.Update(key("-"))
	}
	if a.maxTokens != 0 {
		t.Errorf("maxTokens is %d, want it clamped at 0", a.maxTokens)
	}
	for i := 0; i < 50; i++ {
		a.Update(key("+"))
	}
	if a.maxTokens != 32 {
		t.Errorf("maxTokens is %d, want it clamped at 32", a.maxTokens)
	}
}

func TestInspectCtrlCQuitsFromAnyPhase(t *testing.T) {
	for _, p := range []inspectPhase{inspectLoading, inspectEditing, inspectRunning, inspectBrowsing} {
		a, _ := inspectLiveApp("hi", 3)
		a.phase = p
		if _, cmd := a.Update(tea.KeyMsg{Type: tea.KeyCtrlC}); cmd == nil {
			t.Errorf("ctrl+c should quit from phase %v", p)
		}
	}
}

// The generated text is reconstructed from the step labels, so the header can
// show prompt → output without the engine sending it separately.
func TestInspectGeneratedTextFromSteps(t *testing.T) {
	a, _ := inspectLiveApp("The capital of France is", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})
	feed(t, a,
		InspectStep{Label: "prefill", Tr: fixture()},
		InspectStep{Label: " Paris", Tr: fixture()},
		InspectStep{Label: ".", Tr: fixture()},
		InspectRunDone{},
	)

	if got := a.generated(); got != " Paris." {
		t.Errorf("generated() = %q, want %q", got, " Paris.")
	}
	if out := a.View(); !strings.Contains(out, "Paris.") {
		t.Errorf("header should show the generated text:\n%s", out)
	}
}

// --- live tokenization preview ----------------------------------------------

func TestFormatTokenPreview(t *testing.T) {
	got := formatTokenPreview([]string{"The", "_capital", "_of", "_France", "_is"}, 3)
	if strings.Count(got, "|") != 3 {
		t.Errorf("expected 3 separators for 3 tokens plus an ellipsis, got %q", got)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("a truncated preview should end in an ellipsis, got %q", got)
	}
}

func TestInspectPreviewRendersWhileEditing(t *testing.T) {
	a, _ := inspectLiveApp("The capital of France is", 3)
	a.Update(InspectReady{Info: fixture().Header.Config})
	a.Update(InspectPreview{Tokens: []string{"The", "_capital", "_of", "_France", "_is"}})

	out := a.View()
	if !strings.Contains(out, "5 tokens") {
		t.Errorf("expected a token count for the default prompt:\n%s", out)
	}
	if !strings.Contains(out, "The|_capital") {
		t.Errorf("expected a tokenization preview with visible spaces:\n%s", out)
	}
}
