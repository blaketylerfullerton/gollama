package main

import (
	"errors"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

// feed drives the app through a sequence of messages the way bubbletea would,
// re-issuing each returned command so the streaming loop actually advances.
func feed(t *testing.T, a *app, msgs ...tea.Msg) {
	t.Helper()
	for _, m := range msgs {
		if _, cmd := a.Update(m); cmd != nil {
			// Commands that read the channel would block here; the point of
			// feeding manually is to skip them.
			_ = cmd
		}
	}
}

// Before the first step lands there's nothing to render, but the UI still has to
// show something rather than panic on a nil step.
func TestLiveShowsStatusBeforeFirstStep(t *testing.T) {
	ch := make(chan tea.Msg, 4)
	a := newLiveApp(ch, make(chan request, 1), "", 3)
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
	for v := view(0); v < numViews; v++ {
		a.view = v
		if a.View() == "" {
			t.Errorf("view %v rendered nothing with no steps", v)
		}
	}
}

func TestLiveAccumulatesSteps(t *testing.T) {
	ch := make(chan tea.Msg, 8)
	a := newLiveApp(ch, make(chan request, 1), "", 3)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})

	feed(t, a,
		statusMsg("prefill…"),
		stepMsg{label: "prefill", tr: fixture()},
		stepMsg{label: " Paris", tr: fixture()},
		stepMsg{label: ",", tr: fixture()},
		doneMsg{},
	)

	if len(a.steps) != 3 {
		t.Fatalf("collected %d steps, want 3", len(a.steps))
	}
	// The view should follow the newest step as it arrives.
	if a.cur != 2 {
		t.Errorf("cur is %d, want 2 (should track the latest step)", a.cur)
	}
	if a.phase != phaseBrowsing {
		t.Errorf("doneMsg should return to browsing, phase is %v", a.phase)
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

func TestStepNavigationClamps(t *testing.T) {
	ch := make(chan tea.Msg, 8)
	a := newLiveApp(ch, make(chan request, 1), "", 3)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	feed(t, a,
		stepMsg{label: "prefill", tr: fixture()},
		stepMsg{label: " Paris", tr: fixture()},
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

func TestLiveErrorIsShown(t *testing.T) {
	ch := make(chan tea.Msg, 4)
	a := newLiveApp(ch, make(chan request, 1), "", 3)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a.Update(errMsg{errors.New("no checkpoint at nowhere/")})

	out := a.View()
	if !strings.Contains(out, "no checkpoint at nowhere/") {
		t.Errorf("expected the error text, got:\n%s", out)
	}
	if !strings.Contains(out, "q to quit") {
		t.Error("an error screen should say how to get out of it")
	}
	if a.phase == phaseRunning {
		t.Error("an error should stop the running state")
	}
}

// Navigation must not run off the end when a later step has fewer layers than
// the one currently selected — decode steps are traced separately and a
// truncated trace is possible.
func TestNavigationSurvivesShorterStep(t *testing.T) {
	small := &trace.Trace{Header: trace.Header{
		Config: trace.ModelInfo{NLayer: 1, NHead: 1},
	}}
	ch := make(chan tea.Msg, 4)
	a := newLiveApp(ch, make(chan request, 1), "", 3)
	a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})

	feed(t, a, stepMsg{label: "big", tr: fixture()})
	a.layer = 2
	a.head = 1

	feed(t, a, stepMsg{label: "small", tr: small})
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

func TestBuildRejectsMissingTraceFile(t *testing.T) {
	if _, err := build("does-not-exist.jsonl", "", "", 0); err == nil {
		t.Error("expected an error for a missing trace file")
	}
}

func TestIsStop(t *testing.T) {
	if !isStop(151645, []int{151645, 151643}) {
		t.Error("should match a stop token")
	}
	if isStop(42, []int{151645}) {
		t.Error("should not match a non-stop token")
	}
	if isStop(42, nil) {
		t.Error("no stop tokens means nothing stops")
	}
}
