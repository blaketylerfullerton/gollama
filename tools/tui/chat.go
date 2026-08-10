package tui

import (
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// The third screen: talk to the model you just picked.
//
// It knows nothing about how a token gets generated — no model.GPT, no
// tokenizer.Tokenizer, same as every other file in this package. What it knows
// is a label for the header and two channels: one it reads streamed text and
// status off of, one it writes what you typed onto. Whoever calls NewChat owns
// the engine and decides what "generate" means; this file only owns the frame
// around it.

// ChatToken is a slice of generated text as it comes off the model — usually
// one token, decoded. It's a string rather than an id: this package doesn't
// have a tokenizer to turn one back into the other.
type ChatToken string

// ChatDone says the current turn finished — end of sequence, or the token
// budget ran out. The screen goes back to accepting input.
type ChatDone struct{}

// ChatErr carries a failure from the engine side. It ends the run: there's no
// cache left to trust after a forward pass fails partway through it.
type ChatErr struct{ Err error }

// ChatStatus is free text shown while there's nothing to stream yet — loading
// the checkpoint, or "thinking" between submitting a prompt and the first
// token landing.
type ChatStatus string

// chatPhase is what the screen is doing, which decides where a keystroke goes.
type chatPhase int

const (
	chatLoading    chatPhase = iota // waiting for ChatStatus/ChatErr off the engine
	chatIdle                        // the input is yours
	chatGenerating                  // a turn is in flight; input is read-only
)

// chatTurn is one exchange: what you typed, and however much of the model's
// reply has arrived so far. model grows in place while a turn is in flight,
// which is what makes the screen a stream rather than a spinner.
type chatTurn struct {
	you   string
	model string
}

// Chat is the bubbletea model for the conversation screen.
type Chat struct {
	label string // what's loaded, for the header — a model name, or "loading…"

	events <-chan tea.Msg
	reqs   chan<- string

	phase  chatPhase
	status string
	err    error

	turns []chatTurn
	input textinput.Model
	vp    viewport.Model

	w, h int
}

var _ tea.Model = (*Chat)(nil)

// NewChat builds the screen. events is read for as long as the program runs;
// reqs is written to once per submitted prompt, so it should be buffered by at
// least one or Update blocks the render loop on a slow engine.
func NewChat(label string, events <-chan tea.Msg, reqs chan<- string, prompt string) *Chat {
	in := textinput.New()
	in.Placeholder = "type anything and press enter"
	in.SetValue(prompt)
	in.CharLimit = 2048
	in.Prompt = "❯ "
	in.Focus()

	return &Chat{
		label:  label,
		events: events,
		reqs:   reqs,
		phase:  chatLoading,
		status: "loading…",
		input:  in,
		vp:     viewport.New(0, 0),
	}
}

func (c *Chat) Init() tea.Cmd { return waitForChat(c.events) }

// waitForChat turns the next message off ch into a bubbletea command, the same
// shape every other live screen in this codebase uses to drain a channel: it
// has to be reissued after every message or the stream stalls after one.
func waitForChat(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return ChatErr{Err: errClosed}
		}
		return msg
	}
}

// errClosed is what a closed events channel is reported as. The engine only
// closes it after a failure it has already sent as a ChatErr, so this is only
// ever seen if that message was somehow missed — a channel closing cleanly
// isn't itself news.
var errClosed = chatClosedErr{}

type chatClosedErr struct{}

func (chatClosedErr) Error() string { return "the engine stopped responding" }

func (c *Chat) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		c.w, c.h = msg.Width, msg.Height
		c.layout()
		return c, nil

	case ChatStatus:
		c.status, c.err = string(msg), nil
		if c.phase == chatLoading {
			c.phase = chatIdle
		}
		return c, waitForChat(c.events)

	case ChatToken:
		if len(c.turns) > 0 {
			c.turns[len(c.turns)-1].model += string(msg)
		}
		c.refresh()
		return c, waitForChat(c.events)

	case ChatDone:
		c.phase, c.status = chatIdle, ""
		c.input.Focus()
		return c, waitForChat(c.events)

	case ChatErr:
		c.err, c.phase, c.status = msg.Err, chatIdle, ""
		return c, waitForChat(c.events)

	case tea.KeyMsg:
		return c.handleKey(msg)
	}
	return c, nil
}

func (c *Chat) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "esc":
		return c, tea.Quit
	case "enter":
		return c, c.submit()
	case "up", "down", "pgup", "pgdown", "ctrl+u", "ctrl+d":
		var cmd tea.Cmd
		c.vp, cmd = c.vp.Update(msg)
		return c, cmd
	}
	if c.phase == chatGenerating {
		// The prompt that started this turn is already on screen and the
		// engine is mid-pass; there is nothing a keystroke could do but corrupt
		// the next submission.
		return c, nil
	}
	var cmd tea.Cmd
	c.input, cmd = c.input.Update(msg)
	return c, cmd
}

// submit sends the input's contents to the engine and starts a new turn to
// stream the reply into. A blank line or a screen that's still loading or
// still generating has nowhere to put another request.
func (c *Chat) submit() tea.Cmd {
	text := strings.TrimSpace(c.input.Value())
	if text == "" || c.phase != chatIdle {
		return nil
	}
	c.turns = append(c.turns, chatTurn{you: text})
	c.input.Reset()
	c.phase, c.status, c.err = chatGenerating, "thinking…", nil
	c.refresh()

	reqs := c.reqs
	return func() tea.Msg { reqs <- text; return nil }
}

// refresh re-renders the transcript into the viewport and follows the bottom,
// so a token landing mid-scroll doesn't leave the reader stranded above it.
func (c *Chat) refresh() {
	c.vp.SetContent(c.transcript())
	c.vp.GotoBottom()
}

func (c *Chat) transcript() string {
	if len(c.turns) == 0 {
		return dimStyle.Render("Nothing sent yet — whatever you type continues from where the model's\n" +
			"context left off, the same way the prompt on the previous screen would have.")
	}
	blocks := make([]string, len(c.turns))
	for i, t := range c.turns {
		you := youStyle.Render("you") + "  " + valueStyle.Render(t.you)
		reply := modelReplyStyle.Render(t.model)
		if reply == "" && i == len(c.turns)-1 && c.phase == chatGenerating {
			reply = dimStyle.Render("…")
		}
		blocks[i] = you + "\n" + modelStyle.Render("model") + "  " + reply
	}
	return strings.Join(blocks, "\n\n")
}

// layout sizes the input and the transcript viewport to the current terminal.
// It's called on every resize rather than computed once, for the same reason
// the picker recomputes its row count on resize: the frame this screen sits in
// is the one every screen shares, and that frame's height changes with the
// window.
func (c *Chat) layout() {
	bar := c.bar()
	inner := max(c.w-2*screenMargin, minSpecsWidth)
	rows := bodyRows(c.h, bar) - 2 // less the title and the blank line under it

	c.input.Width = max(inner-4-lipgloss.Width(c.input.Prompt), 8)
	c.vp.Width = inner - 4 // panel padding, no border cost here — see View
	c.vp.Height = max(rows-4, 3)
	c.refresh()
}

func (c *Chat) View() string {
	bar := c.bar()
	inner := max(c.w-2*screenMargin, minSpecsWidth)

	transcript := panelStyle.Width(inner - 2).Height(c.vp.Height).Render(c.vp.View())

	body := lipgloss.JoinVertical(lipgloss.Left,
		header("GoLlama", c.headerSubtitle(), inner), "",
		transcript, "", c.input.View())

	return screen(c.w, c.h, body, bar)
}

func (c *Chat) headerSubtitle() string {
	if c.err != nil {
		return warnStyle.Render(c.err.Error())
	}
	return "chatting with " + c.label
}

func (c *Chat) bar() string {
	keys := []string{
		keyStyle.Render("enter") + dimStyle.Render(" send"),
		keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
		keyStyle.Render("esc") + dimStyle.Render(" quit"),
	}
	return toolbar(c.w, strings.Join(keys, dimStyle.Render(" · ")), dimStyle.Render(c.status))
}
