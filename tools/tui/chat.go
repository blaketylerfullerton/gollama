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
// around it — including a second tab, since "what did the model attend to"
// is a question about that same stream of tokens, not a different program.

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

// ChatStep describes one generated token for the inspect tab: what the model
// attended to while producing it, and what it would have said next at that
// point. It rides alongside the ChatToken for the same token rather than
// replacing it — the chat tab and the inspect tab are two views of one stream,
// not two separate ones.
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

// chatPhase is what the screen is doing, which decides where a keystroke goes.
type chatPhase int

const (
	chatLoading    chatPhase = iota // waiting for ChatStatus/ChatErr off the engine
	chatIdle                        // the input is yours
	chatGenerating                  // a turn is in flight; input is read-only
)

// chatTab is which half of the screen is showing: the conversation, or what
// the last few tokens actually did inside the model.
type chatTab int

const (
	tabConversation chatTab = iota
	tabInspect
)

func (t chatTab) String() string {
	if t == tabInspect {
		return "inspect"
	}
	return "chat"
}

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

// maxSteps bounds how many tokens' worth of inspect detail are kept. Every
// entry holds a handful of ranked lists, and a long conversation shouldn't
// mean an ever-growing one — only the tail of it is ever on screen anyway.
const maxSteps = 64

// Chat is the bubbletea model for the conversation screen.
type Chat struct {
	label string // what's loaded, for the header — a model name, or "loading…"
	arch  Arch   // its shape, for the memory estimate in the stats bar

	events <-chan tea.Msg
	reqs   chan<- string

	phase  chatPhase
	status string
	err    error
	tab    chatTab

	turns []chatTurn
	steps []ChatStep

	turnStarted  time.Time
	turnTokens   int
	lastTurnRate string // "12 tok in 4.1s (2.9 tok/s)", frozen once a turn ends

	sessionID string    // names this conversation's file under history.Save
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
		c.steps = append(c.steps, msg)
		if over := len(c.steps) - maxSteps; over > 0 {
			c.steps = c.steps[over:]
		}
		c.refresh()
		return c, waitForChat(c.events)

	case ChatDone:
		c.phase, c.status = chatIdle, ""
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
	case "ctrl+c", "esc":
		return c, tea.Quit
	case "tab":
		c.tab = (c.tab + 1) % 2
		c.refresh()
		return c, nil
	case "enter":
		return c, c.submit()
	case "up", "down", "pgup", "pgdown", "ctrl+u", "ctrl+d":
		var cmd tea.Cmd
		c.vp, cmd = c.vp.Update(msg)
		return c, cmd
	}
	if c.phase == chatGenerating || c.tab == tabInspect {
		// Mid-turn, the prompt that started it is already on screen and there's
		// nothing a keystroke could do but corrupt the next submission. On the
		// inspect tab there is no box to type into at all — every other key
		// there is a scroll key, handled above.
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

// setContent re-renders whichever tab is showing into the viewport without
// touching scroll position — for updates like the RAM gauge ticking over that
// change what's on screen but aren't new conversation to follow to the bottom
// of.
func (c *Chat) setContent() {
	var content string
	if c.tab == tabInspect {
		content = c.inspect()
	} else {
		content = c.transcript()
	}
	c.vp.SetContent(c.anchorBottom(content, c.vp.Height))
}

// anchorBottom pads content with leading blank lines so short conversations
// sit against the bottom of the panel, next to the input box, the way Claude
// Code's transcript hugs the prompt instead of floating at the top of an
// otherwise empty pane. Once content fills the panel this is a no-op — the
// padding only ever fills the gap, never trims anything.
//
// When the gap is big enough to hold something, it holds the session stats
// instead of going to waste — the same numbers the toolbar shows, just given
// room to be read rather than squeezed onto one line.
func (c *Chat) anchorBottom(content string, height int) string {
	pad := height - lipgloss.Height(content)
	if pad <= 0 {
		return content
	}
	stats := c.sessionStats()
	if statH := lipgloss.Height(stats); pad >= statH+1 {
		above := pad - statH
		return strings.Repeat("\n", above/2) + stats + strings.Repeat("\n", pad-above/2-statH) + content
	}
	return strings.Repeat("\n", pad) + content
}

// sessionStats is the block shown in the empty space above a short
// conversation: the same "is this still fine" numbers as the stats line, plus
// a RAM gauge and the current stage, laid out with room to breathe since
// there's nothing else competing for that space yet.
func (c *Chat) sessionStats() string {
	rows := []string{
		heading("session"),
		"",
		c.ramGauge(),
		row("resident", "~"+sysinfo.Bytes(c.arch.ResidentBytes())),
		row("stage", c.stageLabel()),
	}
	if rate := c.tpsLabel(); rate != "" {
		rows = append(rows, row("tok/s", rate))
	}
	return lipgloss.NewStyle().Align(lipgloss.Center).Render(strings.Join(rows, "\n"))
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
// it fills, no legend needed.
func (c *Chat) ramGauge() string {
	if c.sys.MemoryBytes == 0 {
		return dimStyle.Render("ram unknown")
	}
	used := c.sys.MemoryBytes - c.sys.AvailableBytes
	frac := float64(used) / float64(c.sys.MemoryBytes)
	style := lipgloss.NewStyle().Foreground(amber.Of(frac))

	const width = 20
	filled := min(int(frac*width+0.5), width)
	bar := style.Render(strings.Repeat("█", filled)) +
		amber.Fg(amber.Ash).Render(strings.Repeat("░", width-filled))
	return bar + "  " + dimStyle.Render(memPhrase(c.sys))
}

func (c *Chat) transcript() string {
	if len(c.turns) == 0 {
		return dimStyle.Render("Nothing sent yet — whatever you type continues from where the model's\n" +
			"context left off, the same way the prompt on the previous screen would have.")
	}
	blocks := make([]string, len(c.turns))
	for i, t := range c.turns {
		reply := t.model
		if reply == "" && i == len(c.turns)-1 && c.phase == chatGenerating {
			blocks[i] = renderTurn(t.you, "") + dimStyle.Render("…")
			continue
		}
		blocks[i] = renderTurn(t.you, reply)
	}
	return strings.Join(blocks, "\n\n")
}

// renderTurn is one you/model exchange, styled the same way whether it's
// live in the chat tab or read back from history.Save later — a saved
// conversation should look like the one that produced it.
func renderTurn(you, model string) string {
	return youStyle.Render("you") + "  " + valueStyle.Render(you) + "\n" +
		modelStyle.Render("model") + "  " + modelReplyStyle.Render(model)
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
		turns[i] = history.Entry{You: t.you, Model: t.model}
	}
	_ = history.Save(history.Conversation{
		ID: c.sessionID, Label: c.label, StartedAt: c.startedAt, Turns: turns,
	})
}

// inspect is the second tab: one block per recent token, each with the two
// rankings that explain it — what it leaned on, and what it thought came next.
// It's the same idea as cmd/inspect's logit lens and attention views, sized
// down to fit beside a conversation instead of a whole screen.
func (c *Chat) inspect() string {
	if len(c.steps) == 0 {
		return dimStyle.Render("Nothing generated yet — send something on the chat tab, then come back " +
			"here to see what each token attended to and what it ranked as likely to follow.")
	}
	blocks := make([]string, len(c.steps))
	for i, s := range c.steps {
		blocks[i] = keyStyle.Render(fmt.Sprintf("%q", s.Token)) + "\n" +
			"  " + dimStyle.Render("attended to  ") + candidateList(s.Attention) + "\n" +
			"  " + dimStyle.Render("ranked next  ") + candidateList(s.Candidates)
	}
	return strings.Join(blocks, "\n\n")
}

// candidateList renders a ranked list as "text 42% · text 18% · …", the same
// compact form the picker uses for its own verdicts — a row you can scan
// without the columns of a table.
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
	// Title, blank line, tab strip, blank line, and the stats line above the
	// toolbar all come out of the body before the panel gets what's left.
	rows := bodyRows(c.h, bar) - 4

	c.input.Width = max(inner-4-lipgloss.Width(c.input.Prompt), 8)
	// The panel renders at panelStyle.Width(inner-2), and that call's own
	// padding (4) sits on top of the width it's given rather than inside it —
	// so the text column inside the border and padding is inner-2-4. See
	// picker.go's list/mem panels for the same arithmetic.
	c.vp.Width = inner - 6
	c.vp.Height = max(rows-4, 3)
	c.refresh()
}

func (c *Chat) View() string {
	bar := c.bar()
	inner := max(c.w-2*screenMargin, minSpecsWidth)

	panel := panelStyle.Width(inner - 2).Height(c.vp.Height).Render(c.vp.View())

	rows := []string{header("GoLlama", c.headerSubtitle(), inner), "", c.tabs(inner), "", panel}
	if c.tab == tabConversation {
		rows = append(rows, "", c.input.View())
	}
	rows = append(rows, "", c.stats(inner))

	return screen(c.w, c.h, lipgloss.JoinVertical(lipgloss.Left, rows...), bar)
}

// tabs is the nav strip: which of the two views is showing, and the key that
// switches. It sits under the title rather than in the toolbar because it's a
// choice about the panel below it, not a global command like quitting.
func (c *Chat) tabs(width int) string {
	var cells []string
	for _, t := range []chatTab{tabConversation, tabInspect} {
		label := " " + t.String() + " "
		if t == c.tab {
			cells = append(cells, selectedStyle.Render(label))
		} else {
			cells = append(cells, dimStyle.Render(label))
		}
	}
	strip := strings.Join(cells, dimStyle.Render("│"))
	return strip + dimStyle.Render("   tab to switch")
}

func (c *Chat) headerSubtitle() string {
	if c.err != nil {
		return warnStyle.Render(c.err.Error())
	}
	return "chatting with " + c.label
}

// stats is the line above the toolbar: what this conversation is costing, in
// the same units the picker used to decide whether to load the model at all.
// It's memory rather than the toolbar's keys because it isn't a command — it's
// the answer to "is this still fine", which is worth being able to glance at
// without pressing anything.
//
// Parts drop from the end, least essential first, until what's left fits
// width — the same accommodation the toolbar makes for its own right half. A
// line that wrapped would drag every shorter line above it out to match, since
// lipgloss.JoinVertical pads a block to its widest member.
func (c *Chat) stats(width int) string {
	parts := []string{
		"ram " + memPhrase(c.sys),
		"resident ~" + sysinfo.Bytes(c.arch.ResidentBytes()),
	}
	if c.lastTurnRate != "" {
		parts = append(parts, c.lastTurnRate)
	} else if c.phase == chatGenerating && c.turnTokens > 0 {
		parts = append(parts, fmt.Sprintf("%d tok so far", c.turnTokens))
	}
	for len(parts) > 1 && lipgloss.Width(strings.Join(parts, "   ·   ")) > width {
		parts = parts[:len(parts)-1]
	}
	line := strings.Join(parts, "   ·   ")
	if lipgloss.Width(line) > width {
		line = truncateCells(line, width)
	}
	return dimStyle.Render(line)
}

// truncateCells clips s to n terminal cells, measuring in display width rather
// than bytes — a byte-index slice cuts multi-byte runes like "…" or an em dash
// in half, which renders as a replacement glyph rather than a clean edge.
func truncateCells(s string, n int) string {
	var width int
	for i, r := range s {
		w := lipgloss.Width(string(r))
		if width+w > n {
			return s[:i]
		}
		width += w
	}
	return s
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
		keyStyle.Render("tab") + dimStyle.Render(" inspect"),
		keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
		keyStyle.Render("esc") + dimStyle.Render(" quit"),
	}
	return toolbar(c.w, strings.Join(keys, dimStyle.Render(" · ")), dimStyle.Render(c.status))
}
