package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/history"
)

// newTestChat builds a chat screen with buffered channels sized so a test can
// drive it without a goroutine on the other end pumping messages.
//
// It also redirects history.Save to a temp directory: a ChatDone in these
// tests is a real save, and without this it would write conversations into
// whatever machine happens to be running the suite.
func newTestChat(t *testing.T, prompt string) (*Chat, chan tea.Msg, chan string) {
	history.SetTestDir(t, t.TempDir())
	events := make(chan tea.Msg, 8)
	reqs := make(chan string, 1)
	arch := known[0].Arch // Qwen3-0.6B's real shape, so the stats bar has real numbers to render
	c := NewChat("Qwen3-0.6B", arch, events, reqs, prompt)
	c.Update(tea.WindowSizeMsg{Width: 100, Height: 32})
	return c, events, reqs
}

// Enter while the screen is still loading has nowhere to send a prompt — the
// engine on the other end doesn't have a model loaded yet to run it on.
func TestChatIgnoresSubmitWhileLoading(t *testing.T) {
	c, _, reqs := newTestChat(t, "hello")
	c.Update(tea.KeyMsg{Type: tea.KeyEnter})
	select {
	case req := <-reqs:
		t.Fatalf("a request was sent while still loading: %q", req)
	default:
	}
}

// Once the engine reports itself ready, enter on a non-empty box sends exactly
// what was typed, and the screen stops accepting more until that turn ends.
func TestChatSubmitSendsAndLocksInput(t *testing.T) {
	c, _, reqs := newTestChat(t, "hello there")
	c.Update(ChatStatus(""))
	if c.phase != chatIdle {
		t.Fatalf("phase = %v after ChatStatus, want chatIdle", c.phase)
	}

	_, cmd := c.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("submit produced no command to deliver the request")
	}
	cmd() // the command is what actually performs the channel send

	select {
	case got := <-reqs:
		if got != "hello there" {
			t.Errorf("sent %q, want %q", got, "hello there")
		}
	default:
		t.Fatal("nothing was sent to reqs")
	}
	if c.phase != chatGenerating {
		t.Errorf("phase = %v after submit, want chatGenerating", c.phase)
	}
	if c.input.Value() != "" {
		t.Error("the input was not cleared after submitting it")
	}

	// A second enter mid-turn must not queue a second request — reqs is
	// buffered by exactly one, sized for the request already in flight.
	c.Update(tea.KeyMsg{Type: tea.KeyEnter})
	select {
	case req := <-reqs:
		t.Fatalf("a second request was sent mid-turn: %q", req)
	default:
	}
}

// Tokens streamed in append to the turn that's in flight rather than starting
// a new one, and the transcript has to actually show them — that's the whole
// point of streaming instead of waiting for ChatDone.
func TestChatTokensAppendToTheOpenTurn(t *testing.T) {
	c, _, _ := newTestChat(t, "say hi")
	c.Update(ChatStatus(""))
	c.Update(tea.KeyMsg{Type: tea.KeyEnter})

	c.Update(ChatToken("Hel"))
	c.Update(ChatToken("lo"))
	if got := c.turns[len(c.turns)-1].model; got != "Hello" {
		t.Errorf("turn.model = %q, want %q", got, "Hello")
	}
	if !strings.Contains(c.transcript(), "Hello") {
		t.Error("the streamed text never reached the transcript")
	}

	c.Update(ChatDone{})
	if c.phase != chatIdle {
		t.Errorf("phase = %v after ChatDone, want chatIdle", c.phase)
	}
}

// An engine failure has to end the turn — there's no cache left to trust
// after a forward pass errors partway through it — and it has to say why, or
// the screen just looks stuck.
func TestChatErrEndsTheTurnAndIsShown(t *testing.T) {
	c, _, _ := newTestChat(t, "hello")
	c.Update(ChatStatus(""))
	c.Update(tea.KeyMsg{Type: tea.KeyEnter})

	c.Update(ChatErr{Err: errBoom})
	if c.phase != chatIdle {
		t.Errorf("phase = %v after ChatErr, want chatIdle", c.phase)
	}
	if !strings.Contains(c.View(), "boom") {
		t.Error("the error never reached the screen")
	}
}

var errBoom = boomErr{}

type boomErr struct{}

func (boomErr) Error() string { return "boom" }

// Whatever the terminal size, the frame has to actually fit inside it —
// otherwise every line wraps and the layout falls apart, same requirement as
// the other two screens.
func TestChatFitsTheTerminal(t *testing.T) {
	for _, size := range []tea.WindowSizeMsg{
		{Width: 60, Height: 20},
		{Width: 100, Height: 32},
		{Width: 160, Height: 45},
	} {
		c, _, _ := newTestChat(t, "The capital of France is")
		c.Update(size)
		c.Update(ChatStatus(""))
		c.Update(tea.KeyMsg{Type: tea.KeyEnter})
		c.Update(ChatToken("Paris."))

		lines := strings.Split(c.View(), "\n")
		if len(lines) != size.Height {
			t.Errorf("at %dx%d: %d rows, want %d", size.Width, size.Height, len(lines), size.Height)
		}
		for _, line := range lines {
			if got := lipgloss.Width(line); got > size.Width {
				t.Errorf("at %dx%d: a %d-cell line: %q", size.Width, size.Height, got, line)
			}
		}
	}
}

// The chat screen has no live inspect view of its own — a ChatStep's
// attention and ranking lists should never show up in what's on screen, even
// though the step itself still has to survive onto the turn (see
// TestChatStepsRecordedForHistory) for the past-conversations screen to step
// through later.
func TestChatStepsNotShownLive(t *testing.T) {
	c, _, _ := newTestChat(t, "hello")
	c.Update(ChatStatus(""))
	c.Update(tea.KeyMsg{Type: tea.KeyEnter})
	c.Update(ChatStep{
		Token:      "Paris",
		Attention:  []ChatCandidate{{Text: "capital", Prob: 0.6}},
		Candidates: []ChatCandidate{{Text: ".", Prob: 0.4}},
	})

	if strings.Contains(c.View(), "capital") || strings.Contains(c.View(), "60%") {
		t.Error("inspect content (attention/candidates) leaked into the chat screen's view")
	}
}

// A ChatStep is recorded onto the turn currently in flight so it can be
// written to history — even though nothing on the chat screen itself renders
// it live anymore.
func TestChatStepsRecordedForHistory(t *testing.T) {
	c, _, _ := newTestChat(t, "hello")
	c.Update(ChatStatus(""))
	c.Update(tea.KeyMsg{Type: tea.KeyEnter})
	step := ChatStep{
		Token:      "Paris",
		Attention:  []ChatCandidate{{Text: "capital", Prob: 0.6}},
		Candidates: []ChatCandidate{{Text: ".", Prob: 0.4}},
	}
	c.Update(step)

	if len(c.turns) == 0 || len(c.turns[len(c.turns)-1].steps) != 1 {
		t.Fatalf("turn.steps = %v, want the one ChatStep just sent", c.turns)
	}
	if got := c.turns[len(c.turns)-1].steps[0]; got.Token != step.Token {
		t.Errorf("turn.steps[0].Token = %q, want %q", got.Token, step.Token)
	}
}

// The stats line has real numbers in it whatever the terminal width, and it
// must never be the widest thing on screen — see the width guard in stats().
func TestChatStatsFitsNarrowTerminal(t *testing.T) {
	c, _, _ := newTestChat(t, "hello")
	c.Update(tea.WindowSizeMsg{Width: 50, Height: 20})
	for _, line := range strings.Split(c.View(), "\n") {
		if got := lipgloss.Width(line); got > 50 {
			t.Errorf("a %d-cell line at width 50: %q", got, line)
		}
	}
}

// ctrl+c and esc are the only way out of a screen with no back button, so a
// typo in either leaves the user stuck mid-conversation.
func TestChatQuitKeys(t *testing.T) {
	for _, key := range []tea.KeyMsg{
		{Type: tea.KeyCtrlC},
		{Type: tea.KeyEsc},
	} {
		c, _, _ := newTestChat(t, "hello")
		_, cmd := c.Update(key)
		if cmd == nil {
			t.Errorf("%v did not quit the screen", key)
		}
	}
}
