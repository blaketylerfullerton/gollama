package tui

import (
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

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
// ChatStep is still recorded onto its turn for exactly that. The one exception
// is CommitLayer: the footer status bar reads it straight off the most recent
// step while a turn is in flight (see footerStatus), since a single number
// costs nothing to keep half an eye on and doesn't ask for a tab of its own
// the way the old attention/candidate lists did.

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
	// CommitLayer is how deep into the stack the model's own eventual pick
	// first took the lead in the logit lens, or -1 when this step wasn't
	// traced (the default — see -trace-chat). The footer status bar reads
	// this live, off the most recent step, while a turn is in flight.
	CommitLayer int
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

// ClearMarker is sent on reqs in place of a prompt to ask the engine to drop
// its KV cache and start over, for the /clear command. It's not text any
// prompt could produce, so the engine can tell "start over" apart from an
// ordinary line without the channel needing to carry anything but strings.
const ClearMarker = "\x00gollama:clear\x00"

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
	dir   string // where the weights came from, for the header; empty for the built-in random model

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
	lastCommit   int    // most recent ChatStep.CommitLayer this turn; -1 if none yet (or untraced)

	sessionID string // names this conversation's file under history.Save
	startedAt time.Time

	sys sysinfo.Info

	input textinput.Model
	vp    viewport.Model

	cmdSel int // which of matchingCommands() is highlighted, while / is being typed

	w, h int
}

// chatCommand is one slash command: what to type, and what it does, shown
// side by side in the suggestion row the same way Claude Code's own command
// menu does.
type chatCommand struct {
	name string
	desc string
}

const (
	cmdModel = "/model"
	cmdClear = "/clear"
	cmdHelp  = "/help"
	cmdExit  = "/exit"
)

var chatCommands = []chatCommand{
	{cmdModel, "switch to a different model"},
	{cmdClear, "clear the conversation and free the model's context"},
	{cmdHelp, "list available commands"},
	{cmdExit, "quit gollama"},
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
// showed before this screen, so the two stay honest with each other. dir is
// the checkpoint directory the picker read arch from, shown in the header so
// the screen says where you are the same way it says what you're running.
func NewChat(label string, arch Arch, dir string, events <-chan tea.Msg, reqs chan<- string, prompt string) *Chat {
	in := textinput.New()
	in.Placeholder = "type anything and press enter"
	in.SetValue(prompt)
	in.CharLimit = 2048
	in.Prompt = "❯ "
	in.Focus()

	started := time.Now()
	return &Chat{
		label:      label,
		arch:       arch,
		dir:        dir,
		events:     events,
		reqs:       reqs,
		phase:      chatLoading,
		status:     "loading…",
		sys:        sysinfo.Detect(),
		input:      in,
		vp:         viewport.New(0, 0),
		sessionID:  history.NewID(started),
		startedAt:  started,
		lastCommit: -1,
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
		// The chat screen itself has no inspect view anymore — steps are kept
		// so history.Save has each turn's steps for the past-conversations
		// screen to step through later. CommitLayer is the one piece read
		// live, off the most recent step, for the footer status bar.
		if len(c.turns) > 0 {
			last := &c.turns[len(c.turns)-1]
			last.steps = append(last.steps, msg)
		}
		c.lastCommit = msg.CommitLayer
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

// commandMenuActive says whether the input is in "typing a slash command"
// mode — the input's whole value is a prefix of one, only while there's
// somewhere for it to go. Mid-turn there's no idle input to interpret one
// out of, so it's never active then even if the box still holds a leading
// slash from before the turn started.
func (c *Chat) commandMenuActive() bool {
	return c.phase == chatIdle && strings.HasPrefix(c.input.Value(), "/")
}

// matchingCommands is which commands the current input could still turn
// into. It's every command whose name the input is a prefix of — including
// an exact match, since "/model" is itself a prefix of "/model" — so typing
// a command out in full still leaves exactly one entry highlighted instead
// of none.
func (c *Chat) matchingCommands() []chatCommand {
	v := c.input.Value()
	var out []chatCommand
	for _, cmd := range chatCommands {
		if strings.HasPrefix(cmd.name, v) {
			out = append(out, cmd)
		}
	}
	return out
}

// moveCommandSel steps the highlighted suggestion, wrapping at either end the
// way a short menu should — there's no dead end to bump against with only a
// handful of commands.
func (c *Chat) moveCommandSel(down bool) {
	n := len(c.matchingCommands())
	if n == 0 {
		return
	}
	if down {
		c.cmdSel = (c.cmdSel + 1) % n
	} else {
		c.cmdSel = (c.cmdSel - 1 + n) % n
	}
}

// completeCommand fills the input with the highlighted suggestion's full
// name, the way tab-completion does everywhere else — without running it, so
// a command can still be aimed before it fires.
func (c *Chat) completeCommand() {
	matches := c.matchingCommands()
	if len(matches) == 0 {
		return
	}
	c.input.SetValue(matches[min(c.cmdSel, len(matches)-1)].name)
	c.input.CursorEnd()
	c.cmdSel = 0
}

// runCommand acts on whichever suggestion is highlighted when enter is
// pressed with the menu open. An input that matches nothing — a typo, or a
// slash the user meant as plain text — is reported rather than sent to the
// model: this screen has no way to ask the model to interpret a command it
// was never given a tool to run.
func (c *Chat) runCommand() tea.Cmd {
	matches := c.matchingCommands()
	c.input.Reset()
	if len(matches) == 0 {
		c.err = fmt.Errorf("unknown command — try /help")
		c.refresh()
		return nil
	}
	name := matches[min(c.cmdSel, len(matches)-1)].name
	c.cmdSel = 0
	c.err = nil

	switch name {
	case cmdModel:
		// Same exit as esc: the picker is already what answers "run something
		// else", so switching models is backing out to it, not a screen of
		// its own.
		c.outcome = ChatBack
		return done
	case cmdExit:
		c.outcome = ChatQuit
		return done
	case cmdClear:
		return c.clearConversation()
	case cmdHelp:
		c.showHelp()
		return nil
	}
	return nil
}

// clearConversation drops the visible transcript and starts a fresh history
// entry, then asks the engine to forget its KV cache too — otherwise the
// screen would say the conversation was over while the model kept
// generating as though every turn before /clear were still in its context.
func (c *Chat) clearConversation() tea.Cmd {
	c.turns = nil
	c.lastTurnRate = ""
	c.lastCommit = -1
	c.err = nil
	started := time.Now()
	c.sessionID = history.NewID(started)
	c.startedAt = started
	c.refresh()

	reqs := c.reqs
	return func() tea.Msg { reqs <- ClearMarker; return nil }
}

// showHelp lists the commands as though the model had answered with them —
// same rendering as a real turn, but nothing was sent to the engine and
// nothing here is saved to history, since it isn't part of the conversation.
func (c *Chat) showHelp() {
	var b strings.Builder
	for i, cmd := range chatCommands {
		if i > 0 {
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%-8s %s", cmd.name, cmd.desc)
	}
	c.turns = append(c.turns, chatTurn{you: cmdHelp, model: b.String()})
	c.refresh()
}

func (c *Chat) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	menu := c.commandMenuActive()

	switch msg.String() {
	case "ctrl+c":
		c.outcome = ChatQuit
		return c, done
	case "esc":
		if menu {
			// Cancels the command being typed rather than leaving the
			// screen — esc backing out of chat entirely from here would
			// make "/" itself a trap the moment you change your mind about
			// what to type.
			c.input.Reset()
			c.cmdSel = 0
			return c, nil
		}
		// Back to the picker, the same thing esc does on every other screen —
		// rather than out of the program, which is what it used to do here and
		// what made a mistyped esc mid-conversation unrecoverable.
		c.outcome = ChatBack
		return c, done
	case "enter":
		if menu {
			return c, c.runCommand()
		}
		return c, c.submit()
	case "tab":
		if menu {
			c.completeCommand()
			return c, nil
		}
	case "up", "down":
		if menu {
			c.moveCommandSel(msg.String() == "down")
			return c, nil
		}
		fallthrough
	case "pgup", "pgdown", "ctrl+u", "ctrl+d":
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
	c.cmdSel = 0
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
	c.lastCommit = -1
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

	// The input box is what carries the border now (see View() and
	// inputBoxStyle) — border (2) plus its padding (4) is 6 columns of
	// non-content, the same subtraction the transcript used to make for its
	// own box. See picker.go's list/mem panels for the same arithmetic.
	c.input.Width = max(inner-6-lipgloss.Width(c.input.Prompt), 8)
	// The transcript itself is unboxed now — just its padding (4), no border.
	c.vp.Width = inner - 4
	// rows is the room the transcript and everything under it have to share.
	// Four of those rows aren't viewport: the command hint line, then the
	// input box's own top border, bottom border, and the input line between
	// them. Undercounting this let a box render taller than its budget, and
	// since screen() clips the body from the bottom, the rows it pushed off
	// were the ones at the very end — the chat screen lost its input box at
	// every terminal size.
	c.vp.Height = max(rows-4, 3)
	c.refresh()
}

// inputBoxStyle is panelStyle borrowed for the input line instead of the
// transcript — the box moved from around the conversation to around the one
// thing you're actively doing on this screen, so what you're mid-sentence in
// is what's framed, not what you've already said.
var inputBoxStyle = panelStyle

func (c *Chat) View() string {
	bar := c.bar()
	inner := max(c.w-2*screenMargin, minSpecsWidth)

	transcript := lipgloss.NewStyle().Padding(0, 2).Render(c.vp.View())
	input := inputBoxStyle.Width(inner - 2).Render(c.input.View())

	rows := []string{
		c.infoBox(inner), "",
		transcript, c.commandHint(inner),
		input,
	}

	return screen(c.w, c.h, lipgloss.JoinVertical(lipgloss.Left, rows...), bar)
}

// infoBox is the identity card every chat opens with — the same job a coding
// agent's own splash box does: name the program, say what it's for, and
// place you (which model, which machine, which files) before you type
// anything. It doesn't change once the screen is up, unlike the old corner
// box it replaced, which packed in the stage/ram/tok-s numbers that now live
// in the toolbar instead (see footerStatus) — those move on every token, and
// a card that's supposed to orient you shouldn't be reflowing under your eye
// while you read it.
func (c *Chat) infoBox(available int) string {
	// Less the border, which lipgloss draws outside the width it's given — the
	// panel below does the same subtraction.
	width := available - 2
	// Less the panel's own padding too — panelStyle.Width sets the box's
	// padded interior, so a row's actual text column is 4 narrower than
	// width. titleRule fills every column it's given with rule or badge, so
	// handing it the padded width instead of the content width ran the line
	// 4 columns over and wrapped it.
	content := width - 4
	rows := []string{
		titleRule(fmt.Sprintf("GoLlama %s", Version), content),
		valueStyle.Render("Welcome — type anything to chat, or / for commands"),
		dimStyle.Render("model ") + valueStyle.Render(c.label) +
			dimStyle.Render("   host ") + valueStyle.Render(c.sys.Host),
		dimStyle.Render("dir ") + valueStyle.Render(c.dirLabel()),
	}
	if c.err != nil {
		rows = append(rows, "", warnStyle.Render(c.err.Error()))
	}
	// No blank spacer rows: at a short terminal this box competes with the
	// transcript panel and the input box for the same handful of lines (see
	// layout, and TestChatKeepsTheInputBox), and a header that grows with
	// blank lines takes those rows straight from the input box rather than
	// from anything the user would notice missing.
	return panelStyle.Width(width).Render(strings.Join(rows, "\n"))
}

// dirLabel is where the weights this screen is talking to came from — the
// checkpoint directory, standing in for "the file you're in" the way a
// coding agent's own splash box shows its working directory. The built-in
// demo model was never read off disk, so there's no path to show for it.
func (c *Chat) dirLabel() string {
	if c.dir == "" {
		return "— (built-in random model, no checkpoint on disk)"
	}
	return c.dir
}

// commandHint takes the place of the blank line between the transcript panel
// and the input while a slash command is being typed, listing whichever
// commands it could still become. It's exactly one line whether it has
// anything to say or not — an empty string is one blank line the same as the
// line it replaces — so a command menu opening never resizes anything else
// on screen the way a dropdown panel of its own would.
func (c *Chat) commandHint(width int) string {
	if !c.commandMenuActive() {
		return ""
	}
	matches := c.matchingCommands()
	if len(matches) == 0 {
		return warnStyle.Render("no matching command — try /help")
	}
	sel := min(c.cmdSel, len(matches)-1)
	parts := make([]string, len(matches))
	for i, cmd := range matches {
		if i == sel {
			parts[i] = selectedStyle.Render(" " + cmd.name + " ")
		} else {
			parts[i] = dimStyle.Render(cmd.name)
		}
	}
	line := strings.Join(parts, "  ")
	if desc := matches[sel].desc; lipgloss.Width(line)+3+lipgloss.Width(desc) <= width {
		line += dimStyle.Render("   " + desc)
	}
	return line
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
		keyStyle.Render("/") + dimStyle.Render(" commands"),
		keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
		keyStyle.Render("esc") + dimStyle.Render(" back"),
		keyStyle.Render("ctrl+c") + dimStyle.Render(" quit"),
	}
	return toolbar(c.w, strings.Join(keys, dimStyle.Render(" · ")), dimStyle.Render(c.footerStatus()))
}

// footerStatus is the toolbar's right side. c.status carries the two
// messages that outrank everything else — "loading…" while the checkpoint
// is still coming off disk, "thinking…" mid-turn — and once neither applies,
// this is where the resident-memory and last-turn-rate numbers that used to
// live in the header moved to: still one glance away, just off the identity
// card at the top that no longer changes underneath you turn to turn.
func (c *Chat) footerStatus() string {
	if c.status != "" {
		status := c.status
		if c.phase == chatGenerating {
			if commit := c.commitLabel(); commit != "" {
				status += "  ·  " + commit
			}
		}
		return status
	}
	status := "resident ~" + sysinfo.Bytes(c.arch.ResidentBytes()) + "  ·  ram " + memPhrase(c.sys)
	if rate := c.tpsLabel(); rate != "" {
		status += "  ·  " + rate
	}
	return status
}

// commitLabel is "layer 14/28" — how deep the most recent token needed to go
// before the model's own logit lens agreed with what it actually picked.
// Empty unless -trace-chat is on, since that's what makes CommitLayer
// anything but -1. Kept to a couple of words rather than a fuller phrase:
// toolbar (see layout.go) drops its whole right side, status text included,
// the moment it and the key hints together don't fit one row — a longer
// label would risk taking "thinking…" down with it on an ordinary terminal.
func (c *Chat) commitLabel() string {
	if c.lastCommit < 0 || c.arch.NLayer <= 0 {
		return ""
	}
	return fmt.Sprintf("layer %d/%d", c.lastCommit, c.arch.NLayer)
}
