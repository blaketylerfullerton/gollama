package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/history"
)

// The page behind the welcome menu's third item: every conversation that's
// been saved, read-only. It doesn't reload a model or reopen the engine —
// that's what picking a model from the menu is for — it just plays back what
// was said, the same way scrolling up in a terminal would if the terminal
// hadn't been closed since.

// HistoryOutcome is how the page ended.
type HistoryOutcome int

const (
	// HistoryBack means they backed out to the menu.
	HistoryBack HistoryOutcome = iota
	// HistoryQuit means they left the program entirely.
	HistoryQuit
)

// History is the bubbletea model for the "past conversations" page.
type History struct {
	convos  []history.Conversation
	cursor  int
	viewing bool // false: picking a conversation; true: reading one

	vp      viewport.Model
	outcome HistoryOutcome
	w, h    int
}

var _ tea.Model = (*History)(nil)

// NewHistory loads whatever's been saved. A read failure is treated the same
// as nothing saved yet — the page has one job, and a broken history file
// shouldn't be the reason it can't be opened.
func NewHistory() *History {
	convos, _ := history.List()
	return &History{convos: convos, vp: viewport.New(0, 0), w: 100, h: 32}
}

// Outcome reports how the page ended. Valid once the program has returned.
func (h *History) Outcome() HistoryOutcome { return h.outcome }

func (h *History) Init() tea.Cmd { return nil }

func (h *History) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		h.w, h.h = msg.Width, msg.Height
		h.layout()
		return h, nil

	case tea.KeyMsg:
		if h.viewing {
			return h.updateViewing(msg)
		}
		return h.updateList(msg)
	}
	return h, nil
}

func (h *History) updateList(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "up", "k":
		h.cursor = max(h.cursor-1, 0)
	case "down", "j":
		h.cursor = min(h.cursor+1, len(h.convos)-1)
	case "enter", " ":
		if len(h.convos) == 0 {
			return h, nil
		}
		h.viewing = true
		h.vp.SetContent(h.transcript())
		h.vp.GotoTop()
	case "b", "backspace", "esc":
		h.outcome = HistoryBack
		return h, tea.Quit
	case "q", "ctrl+c":
		h.outcome = HistoryQuit
		return h, tea.Quit
	}
	return h, nil
}

func (h *History) updateViewing(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "b", "backspace", "esc":
		h.viewing = false
		return h, nil
	case "q", "ctrl+c":
		h.outcome = HistoryQuit
		return h, tea.Quit
	default:
		var cmd tea.Cmd
		h.vp, cmd = h.vp.Update(msg)
		return h, cmd
	}
}

func (h *History) layout() {
	bar := h.bar()
	inner := max(h.w-2*screenMargin, minSpecsWidth)
	rows := bodyRows(h.h, bar) - 2 // less the title and the blank line under it

	h.vp.Width = inner - 6
	h.vp.Height = max(rows-2, 3)
	if h.viewing {
		h.vp.SetContent(h.transcript())
	}
}

// transcript renders the highlighted conversation in full, one block per
// turn, the same shape the chat screen itself draws — a saved conversation
// should read like the one that produced it.
func (h *History) transcript() string {
	c := h.convos[h.cursor]
	if len(c.Turns) == 0 {
		return dimStyle.Render("This conversation has no turns in it.")
	}
	blocks := make([]string, len(c.Turns))
	for i, t := range c.Turns {
		blocks[i] = renderTurn(t.You, t.Model)
	}
	return strings.Join(blocks, "\n\n")
}

// list renders every saved conversation as one row: when it started, who it
// was with, and how much was said — enough to tell two sessions with the same
// model apart without opening either.
func (h *History) list() string {
	if len(h.convos) == 0 {
		return dimStyle.Render("Nothing saved yet — conversations are written here once you've sent\n" +
			"and received at least one message on the chat screen.")
	}
	rows := make([]string, len(h.convos))
	for i, c := range h.convos {
		when := c.StartedAt.Format("Jan 2 15:04")
		turns := fmt.Sprintf("%d turn", len(c.Turns))
		if len(c.Turns) != 1 {
			turns += "s"
		}
		line := fmt.Sprintf("%-16s %-24s %s", when, c.Label, turns)
		if i == h.cursor {
			rows[i] = selectedStyle.Render("▸ " + line)
			continue
		}
		rows[i] = "  " + dimStyle.Render(line)
	}
	return strings.Join(rows, "\n")
}

func (h *History) View() string {
	bar := h.bar()
	inner := max(h.w-2*screenMargin, minSpecsWidth)

	subtitle := "past conversations"
	var body string
	if h.viewing {
		c := h.convos[h.cursor]
		subtitle = c.Label + " · " + c.StartedAt.Format("Jan 2 15:04")
		body = h.vp.View()
	} else {
		body = h.list()
	}

	panel := stretch(panelStyle.Width(inner-2), h.vp.Height+2, body)
	return screen(h.w, h.h, lipgloss.JoinVertical(lipgloss.Left,
		header("GoLlama", subtitle, inner), "", panel), bar)
}

func (h *History) bar() string {
	if h.viewing {
		keys := []string{
			keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
			keyStyle.Render("b") + dimStyle.Render(" back to list"),
			keyStyle.Render("q") + dimStyle.Render(" quit"),
		}
		return toolbar(h.w, strings.Join(keys, dimStyle.Render(" · ")), "")
	}
	keys := []string{
		keyStyle.Render("↑↓") + dimStyle.Render(" choose"),
		keyStyle.Render("enter") + dimStyle.Render(" open"),
		keyStyle.Render("b") + dimStyle.Render(" back"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	return toolbar(h.w, strings.Join(keys, dimStyle.Render(" · ")), "")
}
