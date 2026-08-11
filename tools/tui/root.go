package tui

import (
	"context"
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// Root is the whole program: one bubbletea model that owns every screen and
// decides which one is visible.
//
// It used to be a loop that ran a separate tea.Program per screen, each on its
// own alternate screen, returning a value the loop switched on. That worked, but
// it made three things impossible. A screen could only ever be left by ending
// its program, so chat — launched last, from main — had nothing above it to
// return to and its only exit was exiting the process. Nothing could be shared
// between screens, so the terminal size was rediscovered from scratch every
// time: each new model started at its own hardcoded default and snapped to the
// real size when the first WindowSizeMsg arrived, which is the narrow-layout
// flash you saw on every transition. And there was nowhere to put anything that
// spans screens, since there was no model that outlived one.
//
// So the screens stay exactly what they were — bubbletea models that record an
// outcome and say when they're finished — and this sits above them. The only
// change asked of each one is that it returns done instead of tea.Quit: under
// one program a screen returning tea.Quit would take the application with it.
type Root struct {
	checkpointDir string
	root          string // where the catalog looks for checkpoints
	prompt        string // the -prompt flag, prefilled into the chat input
	sys           sysinfo.Info
	engine        Engine

	at      screenID
	welcome *Welcome
	picker  *Picker
	about   *About
	history *History
	chat    *Chat

	// chatStop ends the engine goroutine behind the chat screen. Held here
	// rather than on the Chat, because it has to outlive the screen by exactly
	// as long as it takes to cancel it — see closeChat.
	chatStop context.CancelFunc

	w, h int
}

var _ tea.Model = (*Root)(nil)

// Engine is how the root screen turns a chosen model into a running
// conversation: read prompts off reqs, write ChatToken/ChatStep/ChatDone/ChatErr
// onto events, and stop when ctx is cancelled.
//
// It's a func supplied by the caller rather than something this package does,
// for the same reason no file in here imports engine/model: the screens own the
// frame, and whoever calls Start owns what "generate" means. dir is where the
// weights are, or empty for the built-in random model.
type Engine func(ctx context.Context, dir string, reqs <-chan string, events chan<- tea.Msg)

// screenID names which child is visible.
type screenID int

const (
	atWelcome screenID = iota
	atPicker
	atAbout
	atHistory
	atChat
)

// doneMsg is how a child says it has finished with itself. It replaces the
// tea.Quit each screen used to return: the outcome is already recorded on the
// child (Choice, Outcome), so this carries nothing — the root reads it off
// whichever screen is current.
type doneMsg struct{}

// done is the command every screen returns in place of tea.Quit.
func done() tea.Msg { return doneMsg{} }

// NewRoot builds the program, opening on the welcome screen. The hardware is
// detected once here and handed to every screen that needs it: detection shells
// out to sysctl and vm_stat, and repeating that per screen was a visible pause
// for numbers that cannot have changed.
func NewRoot(checkpointDir, prompt string, engine Engine) *Root {
	sys := sysinfo.Detect()
	return &Root{
		checkpointDir: checkpointDir,
		root:          filepath.Dir(checkpointDir),
		prompt:        prompt,
		sys:           sys,
		engine:        engine,
		at:            atWelcome,
		welcome:       NewWelcomeFor(sys, checkpointDir),
	}
}

// Start puts the whole program on one alternate screen and returns when the user
// leaves it.
func Start(checkpointDir, prompt string, engine Engine) error {
	_, err := tea.NewProgram(NewRoot(checkpointDir, prompt, engine), tea.WithAltScreen()).Run()
	return err
}

func (r *Root) Init() tea.Cmd { return r.welcome.Init() }

func (r *Root) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	// The size is remembered rather than only forwarded, so a screen opened
	// later can be told it before it ever renders. See show.
	if size, ok := msg.(tea.WindowSizeMsg); ok {
		r.w, r.h = size.Width, size.Height
	}
	if _, ok := msg.(doneMsg); ok {
		return r, r.advance()
	}
	return r, r.forward(msg)
}

func (r *Root) View() string { return r.current().View() }

// current is the visible child. Chat is the one that can be absent — it only
// exists while there's an engine behind it — so a message arriving after it was
// closed falls back to the screen that replaced it rather than panicking.
func (r *Root) current() tea.Model {
	switch r.at {
	case atPicker:
		return r.picker
	case atAbout:
		return r.about
	case atHistory:
		return r.history
	case atChat:
		if r.chat != nil {
			return r.chat
		}
	}
	return r.welcome
}

// forward hands a message to the visible child. Everything a screen already
// understood — keys, its own ticks, tokens off the engine — reaches it
// unchanged, which is what let the screens themselves stay as they were.
func (r *Root) forward(msg tea.Msg) tea.Cmd {
	_, cmd := r.current().Update(msg)
	return cmd
}

// advance reads the outcome off the screen that just finished and goes wherever
// it says. This is the switch the old loop in flow.go performed between
// programs; the difference is that it no longer has to end one to run the next,
// which is what gives chat somewhere to go back to.
func (r *Root) advance() tea.Cmd {
	switch r.at {
	case atWelcome:
		switch r.welcome.Choice() {
		case Run:
			return r.show(atPicker)
		case ShowAbout:
			return r.show(atAbout)
		case ShowHistory:
			return r.show(atHistory)
		}

	case atPicker:
		switch r.picker.Outcome() {
		case Selected:
			return r.openChat(r.picker.Selection())
		case Back:
			return r.show(atWelcome)
		}

	case atAbout:
		if r.about.Outcome() == AboutBack {
			return r.show(atWelcome)
		}

	case atHistory:
		if r.history.Outcome() == HistoryBack {
			return r.show(atWelcome)
		}

	case atChat:
		back := r.chat.Outcome() == ChatBack
		r.closeChat()
		if back {
			// Back to the picker rather than the welcome menu: the question chat
			// leaves you with is "run a different one", and that's the screen
			// that answers it — with the cursor still on what you just ran.
			return r.show(atPicker)
		}
	}
	return tea.Quit
}

// show makes to the visible screen, building it if it isn't there yet.
//
// Every screen is handed the current terminal size before it renders a single
// frame. That's the whole reason transitions used to flash: a fresh
// tea.Program's model starts at its own hardcoded default — 100×32 for every
// screen in this package — and only learns the truth when bubbletea sends the
// first WindowSizeMsg, one frame later.
//
// Init is only issued for a screen that was just built. Re-initialising a
// screen that already exists would start a second copy of any ticker it owns,
// and since each tick reissues itself, returning to the welcome screen twice
// would leave the llama animating at double speed and climbing.
func (r *Root) show(to screenID) tea.Cmd {
	var initialize tea.Cmd
	switch to {
	case atWelcome:
		// Reused, so backing out lands on the row you left rather than the top
		// of the menu — but what it says about the world is re-read, since a
		// conversation may have been saved and weights may have been downloaded
		// since it was last on screen.
		r.welcome.reopen()
	case atPicker:
		if r.picker == nil {
			r.picker = NewPicker(r.root, r.sys)
			initialize = r.picker.Init()
		}
	case atAbout:
		if r.about == nil {
			r.about = NewAbout()
			initialize = r.about.Init()
		}
	case atHistory:
		// Rebuilt rather than reused: it reads the whole list up front, and a
		// conversation saved since the last visit has to appear in it.
		r.history = NewHistory()
		initialize = r.history.Init()
	}
	r.at = to
	return tea.Batch(initialize, r.sizeCurrent())
}

// openChat starts an engine for m and shows the conversation.
//
// The checkpoint is not loaded here. m.Name and m.Arch are already known from
// the catalog, so the screen can be on screen saying "loading…" while the
// engine goroutine does the multi-second read behind it — rather than the
// program appearing to hang between two screens with nothing drawn.
func (r *Root) openChat(m Model) tea.Cmd {
	ctx, stop := context.WithCancel(context.Background())
	events := make(chan tea.Msg)
	// Buffered by one, sized for the single request the chat screen allows in
	// flight at a time. That's also what makes closeChat safe: a submission
	// still on its way to a stopped engine has somewhere to land instead of
	// blocking a goroutine forever.
	reqs := make(chan string, 1)
	go r.engine(ctx, m.Dir, reqs, events)

	r.chatStop = stop
	r.chat = NewChat(m.Name, m.Arch, events, reqs, r.prompt)
	r.at = atChat
	return tea.Batch(r.chat.Init(), r.sizeCurrent())
}

// closeChat ends the engine behind the chat screen on the way out of it.
//
// Cancelling rather than closing reqs: the chat screen is the only writer, and
// a submission it has already handed to bubbletea may not have run yet, so
// closing the channel from here could be a send on a closed channel. The engine
// closes events itself as it returns, and by then this screen is unmounted, so
// nothing is left reading it.
func (r *Root) closeChat() {
	if r.chatStop != nil {
		r.chatStop()
		r.chatStop = nil
	}
	r.chat = nil
}

// sizeCurrent tells the visible screen how big the terminal is, synchronously,
// so it lays out at the real size before its first View. Nil before bubbletea
// has told us — its own first WindowSizeMsg will do the job.
func (r *Root) sizeCurrent() tea.Cmd {
	if r.w == 0 {
		return nil
	}
	return r.forward(tea.WindowSizeMsg{Width: r.w, Height: r.h})
}
