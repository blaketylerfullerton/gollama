package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/history"
)

// The page behind the welcome menu's third item: every conversation that's
// been saved, read-only. It doesn't reload a model or reopen the engine —
// that's what picking a model from the menu is for — instead it's the same
// idea as cmd/inspect's stepper turned on a saved chat: step through a reply
// one generated token at a time and see what it attended to and ranked next,
// the same instrumentation the live inspect tab showed while it was running.

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

	// stepping is the token-by-token view: the highlighted conversation shown
	// up to playTurn/playToken rather than dumped on screen in full, the same
	// idea as cmd/inspect's stepper — one generated token at a time, with what
	// it attended to and ranked next underneath it. playing is auto-advance
	// through that view on a timer; without it, arrow keys step by hand.
	stepping            bool
	playing             bool
	playTurn, playToken int

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

// historyTickMsg drives a rewatch forward one step at a time. It's reissued
// with a fresh delay after every tick rather than run on a fixed interval, so
// the pacing can track how fast the turn being replayed actually generated —
// see playDelay.
type historyTickMsg struct{}

func historyTick(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(time.Time) tea.Msg { return historyTickMsg{} })
}

func (h *History) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		h.w, h.h = msg.Width, msg.Height
		h.layout()
		return h, nil

	case historyTickMsg:
		if !h.playing {
			return h, nil // playback was paused or stopped since this tick was scheduled
		}
		h.advanceStep()
		h.vp.SetContent(h.playbackContent())
		h.vp.GotoBottom()
		if !h.playing {
			return h, nil
		}
		return h, historyTick(h.playDelay())

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
		h.viewing, h.stepping, h.playing = true, false, false
		h.vp.SetContent(h.transcript())
		h.vp.GotoTop()
	case "b", "backspace", "esc":
		h.outcome = HistoryBack
		return h, done
	case "q", "ctrl+c":
		h.outcome = HistoryQuit
		return h, done
	}
	return h, nil
}

func (h *History) updateViewing(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if h.stepping {
		return h.updateStepping(msg)
	}
	switch msg.String() {
	case "right", "l", "n", " ":
		return h.enterStepping()
	case "b", "backspace", "esc":
		h.viewing = false
		return h, nil
	case "q", "ctrl+c":
		h.outcome = HistoryQuit
		return h, done
	default:
		var cmd tea.Cmd
		h.vp, cmd = h.vp.Update(msg)
		return h, cmd
	}
}

// updateStepping handles keys once the token-by-token view is open: manual
// stepping, toggling auto-play, or backing out to the full transcript.
func (h *History) updateStepping(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "right", "l", "n":
		h.playing = false // an explicit step always pauses auto-play
		h.advanceStep()
		h.vp.SetContent(h.playbackContent())
		h.vp.GotoBottom()
		return h, nil
	case "left", "h", "N":
		h.playing = false
		h.retreatStep()
		h.vp.SetContent(h.playbackContent())
		h.vp.GotoBottom()
		return h, nil
	case "p", " ":
		h.playing = !h.playing
		if h.playing {
			return h, historyTick(h.playDelay())
		}
		return h, nil
	case "b", "backspace", "esc":
		// Back out of the step view to the full transcript first, then to
		// the list on the next esc — one layer at a time.
		h.stepping, h.playing = false, false
		h.vp.SetContent(h.transcript())
		return h, nil
	case "q", "ctrl+c":
		h.outcome = HistoryQuit
		return h, done
	}
	return h, nil
}

// enterStepping opens the token-by-token view at the very first token of the
// highlighted conversation, paused — stepping through by hand is the default;
// auto-play is opt-in with p or space.
func (h *History) enterStepping() (tea.Model, tea.Cmd) {
	if len(h.convos) == 0 || len(h.convos[h.cursor].Turns) == 0 {
		return h, nil
	}
	h.stepping = true
	h.playTurn, h.playToken = 0, 0
	h.advanceStep() // the key that opened this view is itself the first step
	h.vp.SetContent(h.playbackContent())
	h.vp.GotoBottom()
	return h, nil
}

// advanceStep reveals one more token of the turn currently open, or moves on
// to the next turn once its tokens are exhausted. A turn saved without
// per-token steps — none exist for entries the chat screen never finished —
// is shown whole in one step rather than not at all.
func (h *History) advanceStep() {
	c := h.convos[h.cursor]
	if h.playTurn >= len(c.Turns) {
		h.playing = false
		return
	}
	t := c.Turns[h.playTurn]
	switch {
	case len(t.Steps) == 0:
		h.playTurn++
		h.playToken = 0
	case h.playToken < len(t.Steps):
		h.playToken++
	default:
		h.playTurn++
		h.playToken = 0
	}
	if h.playTurn >= len(c.Turns) {
		h.playing = false
	}
}

// retreatStep un-reveals one token, or steps back into the previous turn —
// shown fully revealed, at the point it itself finished at — once the current
// turn has nothing left to take back.
func (h *History) retreatStep() {
	if h.playToken > 0 {
		h.playToken--
		return
	}
	if h.playTurn == 0 {
		return
	}
	h.playTurn--
	prev := h.convos[h.cursor].Turns[h.playTurn]
	h.playToken = len(prev.Steps)
}

// playDelay is how long to hold the current frame before revealing the next
// token — the turn's own generation rate, so a slow reply rewatches slowly
// and a fast one rewatches fast, rather than every rewatch ticking at the
// same arbitrary speed.
func (h *History) playDelay() time.Duration {
	const fallback = 120 * time.Millisecond
	c := h.convos[h.cursor]
	if h.playTurn >= len(c.Turns) {
		return fallback
	}
	t := c.Turns[h.playTurn]
	if t.Tokens == 0 || t.Elapsed <= 0 {
		return fallback
	}
	if d := t.Elapsed / time.Duration(t.Tokens); d > 0 {
		return d
	}
	return fallback
}

// playbackContent renders every turn before the one currently streaming in
// full, then the in-progress turn's reply built up to however many of its
// tokens have been revealed so far — plus what that last token leaned on and
// ranked next, the same two lines the live inspect tab shows, so a rewatch
// isn't just the text typed out slowly but the same instrumentation replayed.
func (h *History) playbackContent() string {
	c := h.convos[h.cursor]
	var blocks []string
	for i := 0; i < h.playTurn && i < len(c.Turns); i++ {
		blocks = append(blocks, renderTurn(c.Turns[i].You, c.Turns[i].Model))
	}
	if h.playTurn < len(c.Turns) {
		t := c.Turns[h.playTurn]
		partial := t.Model
		var lastStep *history.Step
		if len(t.Steps) > 0 {
			n := min(h.playToken, len(t.Steps))
			var sb strings.Builder
			for _, s := range t.Steps[:n] {
				sb.WriteString(s.Token)
			}
			partial = sb.String()
			if n > 0 {
				lastStep = &t.Steps[n-1]
			}
		}
		block := renderTurn(t.You, partial)
		if lastStep != nil {
			block += "\n" + dimStyle.Render("  attended to  ") + candidateList(fromHistoryCandidates(lastStep.Attention)) +
				"\n" + dimStyle.Render("  ranked next  ") + candidateList(fromHistoryCandidates(lastStep.Candidates))
		}
		blocks = append(blocks, block)
	}
	if len(blocks) == 0 {
		return dimStyle.Render("…")
	}
	return strings.Join(blocks, "\n\n")
}

// fromHistoryCandidates is historyCandidates in reverse — chat.go converts a
// live ranked list to what gets saved; this turns a saved one back into what
// candidateList already knows how to render, so a rewatch draws its ranked
// lists identically to the live inspect tab.
func fromHistoryCandidates(cs []history.Candidate) []ChatCandidate {
	if len(cs) == 0 {
		return nil
	}
	out := make([]ChatCandidate, len(cs))
	for i, c := range cs {
		out[i] = ChatCandidate{Text: c.Text, Prob: c.Prob}
	}
	return out
}

func (h *History) layout() {
	bar := h.bar()
	inner := max(h.w-2*screenMargin, minSpecsWidth)
	rows := bodyRows(h.h, bar) - 2 // less the title and the blank line under it

	h.vp.Width = inner - 6
	h.vp.Height = max(rows-2, 3)
	switch {
	case h.stepping:
		h.vp.SetContent(h.playbackContent())
	case h.viewing:
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
		if h.stepping {
			subtitle += fmt.Sprintf(" · turn %d/%d", min(h.playTurn+1, len(c.Turns)), len(c.Turns))
		}
		body = h.vp.View()
	} else {
		body = h.list()
	}

	panel := stretch(panelStyle.Width(inner-2), h.vp.Height+2, body)
	return screen(h.w, h.h, lipgloss.JoinVertical(lipgloss.Left,
		header("GoLlama", subtitle, inner), "", panel), bar)
}

func (h *History) bar() string {
	if h.stepping {
		playLabel := " auto-play"
		if h.playing {
			playLabel = " pause"
		}
		keys := []string{
			keyStyle.Render("←→") + dimStyle.Render(" step"),
			keyStyle.Render("p") + dimStyle.Render(playLabel),
			keyStyle.Render("b") + dimStyle.Render(" full transcript"),
			keyStyle.Render("q") + dimStyle.Render(" quit"),
		}
		return toolbar(h.w, strings.Join(keys, dimStyle.Render(" · ")), "")
	}
	if h.viewing {
		keys := []string{
			keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
			keyStyle.Render("→") + dimStyle.Render(" step through"),
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
