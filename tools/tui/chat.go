package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
	"github.com/blaketylerfullerton/GoLlama/tools/history"
	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// The third screen: talk to the model you just picked.
//
// It knows nothing about how a token gets generated — no model.GPT, no
// tokenizer.Tokenizer, same as every other file in this package. What it knows
// is a label for the header and two channels: one it reads streamed text and
// status off of, one it writes what you typed onto. Whoever calls NewChat owns
// the engine and decides what "generate" means; this file only owns the frame
// around it.
//
// It has no live inspect view of its own anymore — "what did the model attend
// to" is answered after the fact instead, by stepping through a saved
// conversation on the past-conversations screen (see history.go). Every
// ChatStep is still recorded onto its turn for exactly that; nothing here
// renders one live.

// ChatToken is a slice of generated text as it comes off the model — usually
// one token, decoded. It's a string rather than an id: this package doesn't
// have a tokenizer to turn one back into the other.
type ChatToken string

// ChatCandidate is one entry in a ranked list — a token and a weight, meaning
// either "probability of being next" or "share of attention", depending on
// which list it's in. The screen doesn't care which; it draws both the same
// way, because a bar that's a quarter as long already says "a quarter as
// likely" without a column telling you what kind of likely.
type ChatCandidate struct {
	Text string
	Prob float64
}

// ChatStep describes one generated token: what the model attended to while
// producing it, and what it would have said next at that point. It rides
// alongside the ChatToken for the same token rather than replacing it — the
// chat screen doesn't render this live, but it's recorded onto the turn so
// history.Save has it for the past-conversations screen's step-through.
type ChatStep struct {
	Token      string
	Attention  []ChatCandidate // which earlier tokens this one leaned on
	Candidates []ChatCandidate // what the model ranked highest to come next
}

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

// ChatOutcome is how the chat screen ended.
//
// It didn't used to have one: chat was the last screen of a chain of separate
// programs, so the only way out of it was out of the program, and esc meant quit
// here while it meant back on every screen before it. Now that every screen is
// one model under one program, this screen has somewhere above it to return to.
type ChatOutcome int

const (
	// ChatBack means they backed out to pick something else to run. The
	// conversation is already saved — see save — so leaving costs nothing.
	ChatBack ChatOutcome = iota
	// ChatQuit means they left the program entirely.
	ChatQuit
)

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
//
// steps and elapsed are only filled in once the turn finishes (see ChatDone
// in Update) — they exist so the whole turn, not just its final text, can be
// written to history and played back later the way it actually happened.
type chatTurn struct {
	you   string
	model string

	steps   []ChatStep
	elapsed time.Duration
}

// Chat is the bubbletea model for the conversation screen.
type Chat struct {
	label string // what's loaded, for the header — a model name, or "loading…"
	arch  Arch   // its shape, for the memory estimate in the stats bar

	events <-chan tea.Msg
	reqs   chan<- string

	phase   chatPhase
	status  string
	err     error
	outcome ChatOutcome

	turns []chatTurn

	turnStarted  time.Time
	turnTokens   int
	lastTurnRate string // "12 tok in 4.1s (2.9 tok/s)", frozen once a turn ends

	sessionID string // names this conversation's file under history.Save
	startedAt time.Time

	sys sysinfo.Info

	input textinput.Model
	vp    viewport.Model

	w, h int
}

var _ tea.Model = (*Chat)(nil)

// chatSysTickMsg drives the stats bar's memory reading. sysinfo.Detect shells
// out to sysctl and vm_stat, cheap enough for once every couple of seconds but
// not for every render — a render happens on every keystroke and every token.
type chatSysTickMsg struct{}

const chatSysInterval = 2 * time.Second

func chatSysTick() tea.Cmd {
	return tea.Tick(chatSysInterval, func(time.Time) tea.Msg { return chatSysTickMsg{} })
}

// NewChat builds the screen. events is read for as long as the program runs;
// reqs is written to once per submitted prompt, so it should be buffered by at
// least one or Update blocks the render loop on a slow engine. arch describes
// what's loaded, purely for the memory estimate — the same numbers the picker
// showed before this screen, so the two stay honest with each other.
func NewChat(label string, arch Arch, events <-chan tea.Msg, reqs chan<- string, prompt string) *Chat {
	in := textinput.New()
	in.Placeholder = "type anything and press enter"
	in.SetValue(prompt)
	in.CharLimit = 2048
	in.Prompt = "❯ "
	in.Focus()

	started := time.Now()
	return &Chat{
		label:     label,
		arch:      arch,
		events:    events,
		reqs:      reqs,
		phase:     chatLoading,
		status:    "loading…",
		sys:       sysinfo.Detect(),
		input:     in,
		vp:        viewport.New(0, 0),
		sessionID: history.NewID(started),
		startedAt: started,
	}
}

// Outcome reports how the screen ended. Valid once it has finished.
func (c *Chat) Outcome() ChatOutcome { return c.outcome }

func (c *Chat) Init() tea.Cmd { return tea.Batch(waitForChat(c.events), chatSysTick()) }

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

	case chatSysTickMsg:
		c.sys = sysinfo.Detect()
		c.setContent() // stats may have changed; don't yank the scroll position to do it
		return c, chatSysTick()

	case ChatStatus:
		c.status, c.err = string(msg), nil
		if c.phase == chatLoading {
			c.phase = chatIdle
		}
		c.refresh()
		return c, waitForChat(c.events)

	case ChatToken:
		if len(c.turns) > 0 {
			c.turns[len(c.turns)-1].model += string(msg)
		}
		c.turnTokens++
		c.refresh()
		return c, waitForChat(c.events)

	case ChatStep:
		// The chat screen itself has no inspect view anymore — this is kept
		// only so history.Save has each turn's steps for the past-
		// conversations screen to step through later.
		if len(c.turns) > 0 {
			last := &c.turns[len(c.turns)-1]
			last.steps = append(last.steps, msg)
		}
		return c, waitForChat(c.events)

	case ChatDone:
		c.phase, c.status = chatIdle, ""
		if len(c.turns) > 0 {
			c.turns[len(c.turns)-1].elapsed = time.Since(c.turnStarted)
		}
		c.lastTurnRate = c.turnRate()
		c.input.Focus()
		c.refresh()
		c.save()
		return c, waitForChat(c.events)

	case ChatErr:
		c.err, c.phase, c.status = msg.Err, chatIdle, ""
		c.refresh()
		return c, waitForChat(c.events)

	case tea.KeyMsg:
		return c.handleKey(msg)
	}
	return c, nil
}

// turnRate formats how the turn that just finished went, for the stats bar. It
// only runs once, on ChatDone, rather than every token — the number is more
// readable settled than jittering with every render.
func (c *Chat) turnRate() string {
	if c.turnTokens == 0 {
		return ""
	}
	elapsed := time.Since(c.turnStarted)
	rate := float64(c.turnTokens) / elapsed.Seconds()
	return fmt.Sprintf("%d tok in %s (%.1f tok/s)", c.turnTokens, elapsed.Round(100*time.Millisecond), rate)
}

func (c *Chat) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		c.outcome = ChatQuit
		return c, done
	case "esc":
		// Back to the picker, the same thing esc does on every other screen —
		// rather than out of the program, which is what it used to do here and
		// what made a mistyped esc mid-conversation unrecoverable.
		c.outcome = ChatBack
		return c, done
	case "enter":
		return c, c.submit()
	case "up", "down", "pgup", "pgdown", "ctrl+u", "ctrl+d":
		var cmd tea.Cmd
		c.vp, cmd = c.vp.Update(msg)
		return c, cmd
	}
	if c.phase == chatGenerating {
		// Mid-turn, the prompt that started it is already on screen and
		// there's nothing a keystroke could do but corrupt the next
		// submission.
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
	c.turnStarted, c.turnTokens = time.Now(), 0
	c.refresh()

	reqs := c.reqs
	return func() tea.Msg { reqs <- text; return nil }
}

// refresh re-renders whichever tab is showing into the viewport and follows
// the bottom, so a token landing mid-scroll doesn't leave the reader stranded
// above it.
func (c *Chat) refresh() {
	c.setContent()
	c.vp.GotoBottom()
}

// setContent re-renders the transcript into the viewport without touching
// scroll position — for updates like the RAM gauge ticking over that change
// what's on screen but aren't new conversation to follow to the bottom of.
func (c *Chat) setContent() {
	c.vp.SetContent(c.anchorBottom(c.transcript(), c.vp.Height))
}

// anchorBottom pads content with leading blank lines so short conversations
// sit against the bottom of the panel, next to the input box, the way Claude
// Code's transcript hugs the prompt instead of floating at the top of an
// otherwise empty pane. Once content fills the panel this is a no-op — the
// padding only ever fills the gap, never trims anything.
func (c *Chat) anchorBottom(content string, height int) string {
	if pad := height - lipgloss.Height(content); pad > 0 {
		return strings.Repeat("\n", pad) + content
	}
	return content
}

// stageLabel names what the engine is doing right now, in the same words the
// toolbar's status line would use — loading, idle, or mid-turn.
func (c *Chat) stageLabel() string {
	switch c.phase {
	case chatLoading:
		return "loading"
	case chatGenerating:
		return "generating"
	default:
		return "idle"
	}
}

// tpsLabel is the live rate while a turn is in flight, or the last completed
// turn's rate once it's settled — never both, since only one is ever the
// current answer to "how fast is this going".
func (c *Chat) tpsLabel() string {
	if c.phase == chatGenerating && c.turnTokens > 0 {
		rate := float64(c.turnTokens) / time.Since(c.turnStarted).Seconds()
		return fmt.Sprintf("%.1f", rate)
	}
	if c.lastTurnRate != "" {
		return c.lastTurnRate
	}
	return ""
}

// ramGauge is a compact bar of used-vs-total memory, the same colour-by-
// fraction bar the picker draws before a model is even loaded — brighter as
// it fills, no legend needed. Narrower than the picker's own gauge: this one
// sits in a small corner box rather than a panel with room to spare.
func (c *Chat) ramGauge() string {
	if c.sys.MemoryBytes == 0 {
		return dimStyle.Render("ram unknown")
	}
	used := c.sys.MemoryBytes - c.sys.AvailableBytes
	frac := float64(used) / float64(c.sys.MemoryBytes)
	style := lipgloss.NewStyle().Foreground(amber.Of(frac))

	const width = 14
	filled := min(int(frac*width+0.5), width)
	bar := style.Render(strings.Repeat("█", filled)) +
		amber.NFg(amber.Rule).Render(strings.Repeat("░", width-filled))
	return bar + "  " + dimStyle.Render(memPhrase(c.sys))
}

func (c *Chat) transcript() string {
	width := max(c.vp.Width, 20)
	if len(c.turns) == 0 {
		empty := "Nothing sent yet — whatever you type continues from where the model's " +
			"context left off, the same way the prompt on the previous screen would have."
		return dimStyle.Render(lipgloss.NewStyle().Width(width).Render(empty))
	}
	blocks := make([]string, len(c.turns))
	for i, t := range c.turns {
		reply := t.model
		if reply == "" && i == len(c.turns)-1 && c.phase == chatGenerating {
			blocks[i] = renderTurn(t.you, "", width) + dimStyle.Render("…")
			continue
		}
		blocks[i] = renderTurn(t.you, reply, width)
	}
	return strings.Join(blocks, "\n\n")
}

// modelMarker stands in for the word "model" — a dot rather than a label,
// the way a coding agent's own transcript marks its turns apart from yours
// without spelling out who's talking on every line.
const modelMarker = "●"

// speakerColumn is wide enough for "you" plus a gap — the model's marker
// pads out to the same width so both speakers' text starts in the same
// column instead of the dot's reply sitting flush against the border.
const speakerColumn = 5

// renderTurn is one you/model exchange, styled the same way whether it's
// live in the chat tab or read back from history.Save later — a saved
// conversation should look like the one that produced it.
//
// width is the viewport's own width. bubbles/viewport has no word-wrapping of
// its own — unlike the panels drawn with lipgloss's Width elsewhere in this
// package, it only ever clips a line to its width character-for-character —
// so a long message handed to it raw runs off the right edge instead of
// reflowing. Wrapping it here, before it reaches SetContent, is the same fix
// about.go's body() uses for the same reason.
func renderTurn(you, model string, width int) string {
	return renderSpeaker("you", youStyle, valueStyle, you, width) + "\n" +
		renderSpeaker(modelMarker, modelStyle, modelReplyStyle, model, width)
}

// renderSpeaker is one label/text row, padded to speakerColumn and wrapped to
// width with any continuation lines indented under the label so a long
// message reads as a paragraph rather than a wall of text sitting flush
// against the left edge.
func renderSpeaker(label string, labelStyle, textStyle lipgloss.Style, text string, width int) string {
	pad := max(speakerColumn-lipgloss.Width(label), 1)
	prefixWidth := lipgloss.Width(label) + pad
	wrapWidth := max(width-prefixWidth, 10)
	lines := strings.Split(lipgloss.NewStyle().Width(wrapWidth).Render(text), "\n")

	out := labelStyle.Render(label) + strings.Repeat(" ", pad) + textStyle.Render(lines[0])
	indent := strings.Repeat(" ", prefixWidth)
	for _, l := range lines[1:] {
		out += "\n" + indent + textStyle.Render(l)
	}
	return out
}

// save persists the conversation so far. It's called once per completed
// turn rather than once at the end, so a conversation that's still open when
// the program exits (or crashes) isn't lost — only whatever turn was still
// in flight is. Failures are silent: history is a convenience, not something
// worth interrupting a chat over.
func (c *Chat) save() {
	if len(c.turns) == 0 {
		return
	}
	turns := make([]history.Entry, len(c.turns))
	for i, t := range c.turns {
		turns[i] = history.Entry{
			You: t.you, Model: t.model,
			Tokens: len(t.steps), Elapsed: t.elapsed,
			Steps: historySteps(t.steps),
		}
	}
	_ = history.Save(history.Conversation{
		ID: c.sessionID, Label: c.label, StartedAt: c.startedAt, Turns: turns,
	})
}

// historySteps converts the inspect data captured during a turn into the
// shape history.Save writes to disk, so a rewatch later has the same
// attention and ranking lists the live inspect tab did.
func historySteps(steps []ChatStep) []history.Step {
	out := make([]history.Step, len(steps))
	for i, s := range steps {
		out[i] = history.Step{
			Token:      s.Token,
			Attention:  historyCandidates(s.Attention),
			Candidates: historyCandidates(s.Candidates),
		}
	}
	return out
}

func historyCandidates(cs []ChatCandidate) []history.Candidate {
	if len(cs) == 0 {
		return nil
	}
	out := make([]history.Candidate, len(cs))
	for i, c := range cs {
		out[i] = history.Candidate{Text: c.Text, Prob: c.Prob}
	}
	return out
}

// candidateList renders a ranked list as "text 42% · text 18% · …", the same
// compact form the picker uses for its own verdicts — a row you can scan
// without the columns of a table. The chat screen no longer has a live view
// of its own that uses this — see history.go's step-through, which is the
// only caller left.
func candidateList(cs []ChatCandidate) string {
	if len(cs) == 0 {
		return dimStyle.Render("—")
	}
	parts := make([]string, len(cs))
	for i, c := range cs {
		parts[i] = valueStyle.Render(fmt.Sprintf("%q", c.Text)) + dimStyle.Render(fmt.Sprintf(" %.0f%%", 100*c.Prob))
	}
	return strings.Join(parts, dimStyle.Render("  ·  "))
}

// layout sizes the input and the transcript viewport to the current terminal.
// It's called on every resize rather than computed once, for the same reason
// the picker recomputes its row count on resize: the frame this screen sits in
// is the one every screen shares, and that frame's height changes with the
// window.
func (c *Chat) layout() {
	bar := c.bar()
	inner := max(c.w-2*screenMargin, minSpecsWidth)
	// The info box's own height varies (an error adds a line, a finished turn
	// adds a tok/s row), so it's measured rather than assumed the way a
	// single title line used to be.
	infoHeight := lipgloss.Height(c.infoBox(inner))
	rows := bodyRows(c.h, bar) - infoHeight - 1 // info box, then a blank line before the panel

	c.input.Width = max(inner-4-lipgloss.Width(c.input.Prompt), 8)
	// The panel renders at panelStyle.Width(inner-2), and that call's own
	// padding (4) sits on top of the width it's given rather than inside it —
	// so the text column inside the border and padding is inner-2-4. See
	// picker.go's list/mem panels for the same arithmetic.
	c.vp.Width = inner - 6
	// rows is the room the panel and everything under it have to share. Four of
	// those rows aren't viewport: the panel's own top and bottom border, then the
	// blank line and the input below it. Counting only the last two let the panel
	// render two rows taller than its budget, and since screen() clips the body
	// from the bottom, the two rows it pushed off were the blank and the input —
	// the chat screen lost its input box at every terminal size.
	c.vp.Height = max(rows-4, 3)
	c.refresh()
}

func (c *Chat) View() string {
	bar := c.bar()
	inner := max(c.w-2*screenMargin, minSpecsWidth)

	panel := panelStyle.Width(inner - 2).Height(c.vp.Height).Render(c.vp.View())

	rows := []string{
		c.infoBox(inner), "",
		panel, "",
		c.input.View(),
	}

	return screen(c.w, c.h, lipgloss.JoinVertical(lipgloss.Left, rows...), bar)
}

// infoBoxWidth is the outer width of the corner box — wide enough for its
// widest line, the RAM gauge plus its free/total readout, without stretching
// across the terminal the way a full-width header used to. See style.go's
// panelChrome: the usable text column inside is this minus 6 (padding plus
// border).
const infoBoxWidth = 60

// infoBox is what used to be spread across a header line and a stats line
// above the toolbar, now one box in the top-left corner: what's loaded, what
// it's doing, and what it's costing, the same shape the welcome screen's
// "This machine" panel uses for the same kind of glance-and-move-on numbers.
//
// available is the full width the screen has to work with; the box only
// takes infoBoxWidth of it unless the terminal itself is narrower, so it
// reads as a corner box rather than another full-width header.
func (c *Chat) infoBox(available int) string {
	rows := []string{
		heading(c.label),
		row("stage", c.stageLabel()),
		labelStyle.Render("ram") + c.ramGauge(),
		row("resident", "~"+sysinfo.Bytes(c.arch.ResidentBytes())),
	}
	if rate := c.tpsLabel(); rate != "" {
		rows = append(rows, row("tok/s", rate))
	}
	if c.err != nil {
		rows = append(rows, "", warnStyle.Render(c.err.Error()))
	}
	// Less the border, which lipgloss draws outside the width it's given — the
	// panel below does the same subtraction. Without it the box renders two
	// cells wider than infoBoxWidth claims, and at a narrow terminal those two
	// cells plus the screen margin put the frame over the edge.
	return panelStyle.Width(min(infoBoxWidth, available) - 2).Render(strings.Join(rows, "\n"))
}

// memPhrase is "6.4GB free of 16.0GB", or "unknown" on a platform sysinfo
// couldn't read — the same fallback the welcome screen uses for the same
// reason: a blank field reads as a bug, and a platform that doesn't report
// memory isn't one.
func memPhrase(s sysinfo.Info) string {
	if s.AvailableBytes == 0 || s.MemoryBytes == 0 {
		return "unknown"
	}
	return sysinfo.Bytes(int64(s.AvailableBytes)) + " free of " + sysinfo.Bytes(int64(s.MemoryBytes))
}

func (c *Chat) bar() string {
	keys := []string{
		keyStyle.Render("enter") + dimStyle.Render(" send"),
		keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
		keyStyle.Render("esc") + dimStyle.Render(" back"),
		keyStyle.Render("ctrl+c") + dimStyle.Render(" quit"),
	}
	return toolbar(c.w, strings.Join(keys, dimStyle.Render(" · ")), dimStyle.Render(c.status))
}
