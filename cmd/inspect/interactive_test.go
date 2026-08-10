package main

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
)

// liveApp wires up an app in live mode with a request channel we can inspect.
func liveApp(prompt string, n int) (*app, chan request) {
	reqs := make(chan request, 4)
	a := newLiveApp(make(chan tea.Msg, 8), reqs, prompt, n)
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 40})
	return a, reqs
}

// realTokenizer loads the checkpoint's tokenizer, or skips.
func realTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.FromDirectory("../../checkpoints/qwen3-0.6b")
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	return tok
}

func TestStartsInLoadingPhase(t *testing.T) {
	a, _ := liveApp("hi", 3)
	if a.phase != phaseLoading {
		t.Errorf("phase is %v, want phaseLoading", a.phase)
	}
	// Keystrokes must not reach the prompt field before the model is ready.
	a.Update(key("x"))
	if got := a.input.Value(); got != "hi" {
		t.Errorf("input changed to %q while still loading", got)
	}
}

// readyMsg is what flips the UI from "loading" to accepting input.
func TestReadyEnablesEditing(t *testing.T) {
	a, _ := liveApp("hi", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})

	if a.phase != phaseEditing {
		t.Fatalf("phase is %v, want phaseEditing", a.phase)
	}
	if !a.input.Focused() {
		t.Error("the prompt field should take focus when the model is ready")
	}
	if out := a.View(); !strings.Contains(out, "enter to run") {
		t.Errorf("editing view should say how to run:\n%s", out)
	}
}

func TestTypingEditsThePrompt(t *testing.T) {
	a, _ := liveApp("", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})

	for _, r := range "cats" {
		a.Update(key(string(r)))
	}
	if got := a.input.Value(); got != "cats" {
		t.Errorf("input is %q, want %q", got, "cats")
	}

	// Navigation keys are text while editing, not commands.
	before := a.layer
	a.Update(key("j"))
	if a.layer != before {
		t.Error("j should type a character while editing, not move the layer")
	}
	if got := a.input.Value(); got != "catsj" {
		t.Errorf("input is %q, want %q", got, "catsj")
	}
}

func TestEnterSubmitsThePrompt(t *testing.T) {
	a, reqs := liveApp("the sky is", 4)
	a.Update(readyMsg{cfg: fixture().Header.Config})

	_, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("enter should produce a command that sends the request")
	}
	cmd() // bubbletea would run this on its own goroutine

	select {
	case req := <-reqs:
		if req.prompt != "the sky is" {
			t.Errorf("submitted %q", req.prompt)
		}
		if req.maxTokens != 4 {
			t.Errorf("maxTokens is %d, want 4", req.maxTokens)
		}
	default:
		t.Fatal("nothing was sent to the engine")
	}

	if a.phase != phaseRunning {
		t.Errorf("phase is %v, want phaseRunning", a.phase)
	}
	if a.input.Focused() {
		t.Error("the field should blur while inference runs")
	}
}

// Submitting again has to discard the previous run, or the step strip would mix
// two prompts together.
func TestSubmitClearsPreviousRun(t *testing.T) {
	a, reqs := liveApp("first", 2)
	a.Update(readyMsg{cfg: fixture().Header.Config})
	feed(t, a,
		stepMsg{label: "prefill", tr: fixture()},
		stepMsg{label: " x", tr: fixture()},
		runDoneMsg{},
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

func TestEmptyPromptIsNotSubmitted(t *testing.T) {
	a, reqs := liveApp("", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})

	_, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd != nil {
		cmd()
	}
	select {
	case req := <-reqs:
		t.Errorf("an empty prompt was submitted: %+v", req)
	default:
	}
	if a.phase != phaseEditing {
		t.Errorf("phase is %v, should still be editing", a.phase)
	}
}

func TestIReturnsToThePrompt(t *testing.T) {
	a, _ := liveApp("hi", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})
	feed(t, a, stepMsg{label: "prefill", tr: fixture()}, runDoneMsg{})

	if a.phase != phaseBrowsing {
		t.Fatalf("phase is %v, want phaseBrowsing", a.phase)
	}
	a.Update(key("i"))
	if a.phase != phaseEditing {
		t.Errorf("i should return to the prompt, phase is %v", a.phase)
	}
	if !a.input.Focused() {
		t.Error("the field should regain focus")
	}
}

// esc from the prompt goes to browsing, but only when there's something to
// browse — otherwise it would leave the user on a dead screen.
func TestEscFromPromptNeedsResults(t *testing.T) {
	a, _ := liveApp("hi", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})

	a.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if a.phase != phaseEditing {
		t.Error("esc with no results should keep the user in the prompt")
	}

	feed(t, a, stepMsg{label: "prefill", tr: fixture()})
	a.phase = phaseEditing
	a.input.Focus()
	a.Update(tea.KeyMsg{Type: tea.KeyEsc})
	if a.phase != phaseBrowsing {
		t.Errorf("esc with results should switch to browsing, phase is %v", a.phase)
	}
}

func TestTokenCountAdjustment(t *testing.T) {
	a, _ := liveApp("hi", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})
	a.phase = phaseBrowsing // +/- are browsing-mode keys

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

func TestCtrlCQuitsFromAnyPhase(t *testing.T) {
	for _, p := range []phase{phaseLoading, phaseEditing, phaseRunning, phaseBrowsing} {
		a, _ := liveApp("hi", 3)
		a.phase = p
		if _, cmd := a.Update(tea.KeyMsg{Type: tea.KeyCtrlC}); cmd == nil {
			t.Errorf("ctrl+c should quit from phase %v", p)
		}
	}
}

// The generated text is reconstructed from the step labels, so the header can
// show prompt → output without the engine sending it separately.
func TestGeneratedTextFromSteps(t *testing.T) {
	a, _ := liveApp("The capital of France is", 3)
	a.Update(readyMsg{cfg: fixture().Header.Config})
	feed(t, a,
		stepMsg{label: "prefill", tr: fixture()},
		stepMsg{label: " Paris", tr: fixture()},
		stepMsg{label: ".", tr: fixture()},
		runDoneMsg{},
	)

	if got := a.generated(); got != " Paris." {
		t.Errorf("generated() = %q, want %q", got, " Paris.")
	}
	if out := a.View(); !strings.Contains(out, "Paris.") {
		t.Errorf("header should show the generated text:\n%s", out)
	}
}

// --- live tokenization preview ----------------------------------------------

// Showing the tokenization as you type is only useful if it's the real one, so
// this runs against the actual vocabulary.
func TestTokenPreviewUsesRealTokenizer(t *testing.T) {
	tok := realTokenizer(t)
	a, _ := liveApp("The capital of France is", 3)
	a.Update(readyMsg{tok: tok, cfg: fixture().Header.Config})

	out := a.View()
	if !strings.Contains(out, "5 tokens") {
		t.Errorf("expected a token count for the default prompt:\n%s", out)
	}
	// Whitespace is made visible, so a leading space in a token is not silent.
	if !strings.Contains(out, "The|_capital") {
		t.Errorf("expected a tokenization preview with visible spaces:\n%s", out)
	}
}

func TestTokenPreviewTruncates(t *testing.T) {
	tok := realTokenizer(t)
	ids := tok.Encode("one two three four five six seven eight nine ten eleven twelve thirteen")
	got := tokenPreview(tok, ids, 3)

	if strings.Count(got, "|") != 3 {
		t.Errorf("expected 3 separators for 3 tokens plus an ellipsis, got %q", got)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("a truncated preview should end in an ellipsis, got %q", got)
	}
}
