// Package tui is the terminal front end for GoLlama.
//
// The first thing it shows is a welcome screen: the llama on the left, the
// machine it's about to run on down the right. That pairing is the point — a
// 0.6B model in scalar Go is entirely at the mercy of the cores and the memory
// listed there, so seeing them before the first token makes the speed that
// follows legible rather than disappointing.
//
// Nothing in here imports model/ or tokenizer/. The screen describes hardware
// and what's on disk; loading a checkpoint is the caller's job, after the user
// has said to go ahead.
package tui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/history"
	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// Choice is what the user did with the welcome screen.
type Choice int

const (
	// Quit means they backed out — ctrl+c or q. The caller should exit
	// without loading anything.
	Quit Choice = iota
	// Run means they picked a tool — go to the picker to choose a model for
	// it. Which tool is in Tool().
	Run
	// ShowAbout means they pressed the about key — go read the about page,
	// then come back to this menu.
	ShowAbout
	// ShowHistory means they pressed the history key — go read saved chats,
	// then come back to this menu.
	ShowHistory
)

// Tool is which tool a Run choice picked. Root reads it once, right after
// Choice()==Run, to decide what the picker leads to.
type Tool int

const (
	// ToolAblation forces a chosen attention head's output to zero and
	// compares the answer against the baseline — the one tool here that
	// intervenes instead of just watching. First in the menu because it's
	// the most distinctive thing GoLlama can show.
	ToolAblation Tool = iota
	ToolAttention
	ToolAttribution
	ToolLens
	// ToolChat talks to the model turn by turn, same as every other tool a
	// peer of in this list rather than the separate program it used to be.
	ToolChat
)

// menuItem is one box of the start menu: what it's called, which tool
// picking it opens, and the detail panel shown while it's highlighted.
type menuItem struct {
	title string
	tool  Tool
	// detail renders the content panel for this item. It's a func rather than
	// a precomputed string because some detail panels need the terminal width
	// to wrap, and that isn't known until View runs.
	detail func(w *Welcome) string
}

var menuItems = []menuItem{
	{title: "Ablation", tool: ToolAblation, detail: (*Welcome).ablationBlurb},
	{title: "Attention", tool: ToolAttention, detail: (*Welcome).attentionBlurb},
	{title: "Attribution", tool: ToolAttribution, detail: (*Welcome).attributionBlurb},
	{title: "Logit Lens", tool: ToolLens, detail: (*Welcome).lensBlurb},
	{title: "Chat", tool: ToolChat, detail: (*Welcome).chatBlurb},
}

// Checkpoint is what we found on disk where the weights are meant to be. It's
// scanned rather than loaded: a directory listing costs nothing, and knowing the
// checkpoint is missing before spending seconds on a load is worth a stat call.
type Checkpoint struct {
	Dir     string
	Present bool
	Bytes   int64
}

// ScanCheckpoint looks for weights in dir. A missing directory is not an error —
// it's the normal state of a fresh clone, and the screen says so.
func ScanCheckpoint(dir string) Checkpoint {
	c := Checkpoint{Dir: dir}
	// Qwen3-0.6B ships as a single model.safetensors; everything bigger
	// shards across model-0000N-of-0000M.safetensors instead, named by
	// model.safetensors.index.json. Checking only the single-file name
	// reported a fully downloaded 1.7B/4B/8B checkpoint as not present at
	// all — this package doesn't import engine/model to ask it directly
	// (see root.go's Engine type), so the check is duplicated here rather
	// than shared.
	_, errSingle := os.Stat(filepath.Join(dir, "model.safetensors"))
	_, errSharded := os.Stat(filepath.Join(dir, "model.safetensors.index.json"))
	if errSingle != nil && errSharded != nil {
		return c
	}
	c.Present = true
	// Sum the top level only. The HuggingFace downloader leaves a .cache tree of
	// hardlinked blobs beside the weights, and walking into it would double every
	// file it duplicates.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return c
	}
	for _, e := range entries {
		info, err := e.Info()
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		c.Bytes += info.Size()
	}
	return c
}

// Welcome is the bubbletea model for the splash screen.
type Welcome struct {
	sys sysinfo.Info
	// dir is kept so reopen can re-scan it. The screen is reused rather than
	// rebuilt every time it comes back into view, and what it found on disk the
	// first time may not still be true.
	dir          string
	ckpt         Checkpoint
	historyCount int // how many saved conversations to mention in the chat blurb
	choice       Choice
	tool         Tool // which menuItems row Choice()==Run picked
	cursor       int  // which menuItems row is highlighted
	w, h         int
	// tick counts animation frames rather than storing one, so the llama's two
	// motions can be derived from it independently. See llamaFrameAt.
	tick int
}

var _ tea.Model = (*Welcome)(nil)

// NewWelcome detects the hardware and scans checkpointDir. Both are done here,
// once, rather than in View — View runs on every keystroke and every resize.
func NewWelcome(checkpointDir string) *Welcome {
	return NewWelcomeFor(sysinfo.Detect(), checkpointDir)
}

// NewWelcomeFor is NewWelcome with the hardware already detected, for callers
// that are going to show more than one screen and shouldn't shell out to sysctl
// once per screen.
func NewWelcomeFor(sys sysinfo.Info, checkpointDir string) *Welcome {
	return &Welcome{
		sys:          sys,
		dir:          checkpointDir,
		ckpt:         ScanCheckpoint(checkpointDir),
		historyCount: history.Count(),
		w:            100, h: 32,
	}
}

// Choice reports what the user picked. Valid once the screen has finished.
func (w *Welcome) Choice() Choice { return w.choice }

// Tool reports which tool a Run choice picked. Valid once the screen has
// finished with Choice()==Run.
func (w *Welcome) Tool() Tool { return w.tool }

// reopen re-reads what the menu describes, for a screen that is about to be
// shown again rather than built.
//
// Both numbers in it go stale the moment you leave: a conversation on the chat
// screen adds one to the count the third box quotes, and weights downloaded
// after reading the command in the first box turn "none found" into a model. The
// choice is cleared too — it's what the root screen reads to decide where a
// keypress leads, and a stale one is a decision nobody made.
func (w *Welcome) reopen() {
	w.ckpt = ScanCheckpoint(w.dir)
	w.historyCount = history.Count()
	w.choice = Quit
}

// Init starts the llama animating and names the terminal tab. Nothing else on
// this screen moves, so if the art is never shown the ticks are wasted — but a
// timer at llamaInterval costs less than deciding per frame whether it's
// needed, and View is the only thing that knows the terminal is too narrow for
// the art.
func (w *Welcome) Init() tea.Cmd {
	return tea.Batch(tea.SetWindowTitle("🦙 GoLlama"), llamaTick())
}

func (w *Welcome) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		w.w, w.h = msg.Width, msg.Height
	case llamaTickMsg:
		w.tick++
		return w, llamaTick()
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			w.cursor = max(w.cursor-1, 0)
		case "down", "j":
			w.cursor = min(w.cursor+1, len(menuItems)-1)
		case "enter", " ":
			w.choice = Run
			w.tool = menuItems[w.cursor].tool
			return w, done
		// a and h are off the row list entirely now that every row is a tool —
		// About and past conversations still need a way in, they just
		// shouldn't compete with tools for a cursor row.
		case "a":
			w.choice = ShowAbout
			return w, done
		case "h":
			w.choice = ShowHistory
			return w, done
		// esc is deliberately not bound here: every other screen uses it to go
		// back a level, and this is the screen you land back on after doing
		// that. Wiring the same key to quit here would mean the one reflex
		// trained everywhere else silently closes the program the first time
		// it's used out of habit.
		case "q", "ctrl+c":
			w.choice = Quit
			return w, done
		}
	}
	return w, nil
}

// minSpecsWidth is the narrowest the panel can be rendered and still read as
// a panel rather than a column of wrapped fragments. Shared with every other
// screen's own minimum-width floor, for the same reason.
const minSpecsWidth = 48

const subtitle = "a transformer you can read, one token at a time"

func (w *Welcome) View() string {
	bar := w.bar()
	rows := bodyRows(w.h, bar)
	inner := max(w.w-2*screenMargin, minSpecsWidth)

	// Side by side when there's room for both, stacked when there isn't, and
	// stacked with no art at all when even that doesn't fit — the llama says
	// nothing the panel doesn't, so it's the first thing to go.
	//
	// Either way the panel takes every column the art leaves, so the layout
	// reaches the right-hand edge of the terminal rather than ending wherever
	// the longest hostname happens to.
	if side := inner - llamaWidth - 2; side >= minSpecsWidth {
		// The llama is centred against the column rather than hung from its
		// top: the box is now as tall as the terminal, and art pinned to the
		// top of it leaves the bottom half of the screen visibly empty.
		body := lipgloss.JoinHorizontal(lipgloss.Center, llama(w.tick), "  ", w.column(side, rows))
		return screen(w.w, w.h, body, bar)
	}

	if rows > llamaHeight+12 && llamaWidth <= inner {
		body := lipgloss.JoinVertical(lipgloss.Left, llama(w.tick), "",
			w.column(inner, rows-llamaHeight-1))
		return screen(w.w, w.h, body, bar)
	}
	return screen(w.w, w.h, w.column(inner, rows), bar)
}

// column is the title with the menu and its detail panel under it, filling
// rows.
//
// The title sits on top of the box rather than off in the corner of the screen.
// It names what the panel beneath it is describing, and a heading with its body
// directly under it reads as one thing, where a heading alone at the far left
// reads as page furniture.
func (w *Welcome) column(width, rows int) string {
	head := hero("GoLlama", subtitle, width)
	machine := stretch(panelStyle.Width(width-2), 0, w.machineSpecs())
	// Whatever's left after the title and the machine panel goes to the menu
	// boxes, so the column still reaches the bottom of the screen.
	menu := w.menu(width, rows-lipgloss.Height(head)-lipgloss.Height(machine)-2)
	return lipgloss.JoinVertical(lipgloss.Left,
		head,
		"",
		machine,
		"",
		menu)
}

// menu is the list of tools, toggled up/down the same way the picker's and
// history's own lists are, with the highlighted one's detail rendered in a
// single panel underneath it.
func (w *Welcome) menu(width, rows int) string {
	titles := make([]string, len(menuItems))
	for i, item := range menuItems {
		if i == w.cursor {
			titles[i] = selectedStyle.Render("▸ " + item.title)
			continue
		}
		titles[i] = "  " + dimStyle.Render(item.title)
	}
	list := strings.Join(titles, "\n")
	panel := stretch(panelStyle.Width(width-2), rows-lipgloss.Height(list)-1,
		menuItems[w.cursor].detail(w))
	return lipgloss.JoinVertical(lipgloss.Left, list, "", panel)
}

// machineSpecs is the always-visible panel above the menu: what this machine
// is and what's on disk to run on it, so every tool below can be judged
// against real hardware rather than in the abstract. Weights status used to
// be its own detail panel under a "select a model" row; now that every row is
// a tool rather than one of them being about models specifically, it lives
// here instead, where it's never lost regardless of which tool is
// highlighted.
func (w *Welcome) machineSpecs() string {
	s := w.sys
	rows := []string{
		heading("This machine"),
		row("host", s.Host),
		row("chip", s.CPU),
		row("cores", s.CoreSummary()),
	}
	if s.GPU != "" {
		rows = append(rows, row("gpu", s.GPU))
	}
	// Free memory is shown beside the total because it's the one that decides
	// whether a checkpoint loads comfortably, and on a machine that's been up
	// for a week the two are nothing like each other.
	memory := row("memory", s.Memory())
	if s.AvailableBytes > 0 {
		memory = styledRow("memory", valueStyle.Render(s.Memory())+
			dimStyle.Render("  "+s.Available()+" free"))
	}
	rows = append(rows,
		memory,
		row("platform", s.Platform()),
		row("runtime", fmt.Sprintf("%s · GOMAXPROCS %d", s.GoVersion, s.GOMAXPROCS)),
		w.weightsRow(),
	)
	return strings.Join(rows, "\n")
}

// weightsRow is one line, not a section of its own: this used to be a detail
// panel shown only while "select a model" was highlighted, with room for a
// few paragraphs of guidance. Folded into the always-visible machine panel it
// has to sit above five tool blurbs rather than replace one of them, so it
// says what's on disk and leaves the how — the picker downloads a missing
// model itself, no separate command — to the picker, which is where that
// actually happens.
func (w *Welcome) weightsRow() string {
	if !w.ckpt.Present {
		// Just the verdict, not the expected path: that can be long enough to
		// wrap onto a second line in this always-visible panel, where the
		// detail panel it used to live in had room to spare for one. The
		// picker — one keypress away, on every tool — says the rest.
		return styledRow("weights", warnStyle.Render("none found"))
	}
	return styledRow("weights", valueStyle.Render(fmt.Sprintf("%s (%s on disk)",
		filepath.Base(w.ckpt.Dir), sysinfo.Bytes(w.ckpt.Bytes))))
}

// ablationBlurb is the detail panel for the menu's first row: what makes
// ablation different from the three tools under it.
func (w *Welcome) ablationBlurb() string {
	return strings.Join([]string{
		heading("Ablation"),
		"",
		dimStyle.Render("Force one attention head's output to zero and re-run the prompt —"),
		dimStyle.Render("if the answer actually moves, the head mattered; if it doesn't,"),
		dimStyle.Render("whatever attribution measured wasn't load-bearing. The one tool"),
		dimStyle.Render("here that intervenes instead of just watching."),
	}, "\n")
}

// attentionBlurb is the detail panel for the menu's second row.
func (w *Welcome) attentionBlurb() string {
	return strings.Join([]string{
		heading("Attention"),
		"",
		dimStyle.Render("See what each head attended to, layer by layer — the causal"),
		dimStyle.Render("weights over every earlier token, coloured by magnitude."),
	}, "\n")
}

// attributionBlurb is the detail panel for the menu's third row.
func (w *Welcome) attributionBlurb() string {
	return strings.Join([]string{
		heading("Attribution"),
		"",
		dimStyle.Render("See which components — which head, which MLP — actually pushed"),
		dimStyle.Render("the final answer, and by how much, positive or negative."),
	}, "\n")
}

// lensBlurb is the detail panel for the menu's fourth row.
func (w *Welcome) lensBlurb() string {
	return strings.Join([]string{
		heading("Logit Lens"),
		"",
		dimStyle.Render("Watch the prediction form layer by layer: what the model would"),
		dimStyle.Render("have said if it stopped early, and where the real answer first"),
		dimStyle.Render("takes the lead."),
	}, "\n")
}

// chatBlurb is the detail panel for the menu's fifth row.
func (w *Welcome) chatBlurb() string {
	rows := []string{
		heading("Chat"),
		"",
		dimStyle.Render("Talk to the model turn by turn. Nothing here is shown live, but"),
		dimStyle.Render("every reply is recorded —"),
	}
	rows = append(rows, dimStyle.Render(fmt.Sprintf("%s history for %s.",
		keyStyle.Render("h"), w.historySummary())))
	return strings.Join(rows, "\n")
}

// historySummary is the one-line count chatBlurb reads off history — how many
// conversations there are to read back, not their contents, which is what the
// history page itself is for.
func (w *Welcome) historySummary() string {
	if w.historyCount == 0 {
		return "past conversations (none saved yet)"
	}
	plural := "s"
	if w.historyCount == 1 {
		plural = ""
	}
	return fmt.Sprintf("%d saved conversation%s", w.historyCount, plural)
}

// bar is the toolbar along the bottom: the keys this screen answers to on the
// left, and the other way into the program on the right.
func (w *Welcome) bar() string {
	keys := []string{
		keyStyle.Render("↑↓") + dimStyle.Render(" choose"),
		keyStyle.Render("enter") + dimStyle.Render(" select"),
		keyStyle.Render("a") + dimStyle.Render(" about"),
		keyStyle.Render("h") + dimStyle.Render(" history"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	return toolbar(w.w, strings.Join(keys, dimStyle.Render(" · ")), "")
}

// --- small formatting helpers ------------------------------------------------

func heading(s string) string { return headingStyle.Render(s) }

// row is a label/value pair with the labels aligned into a column. An empty
// value reads as "unknown" rather than as a blank line, since on an unsupported
// platform half these fields legitimately come back empty.
func row(label, value string) string {
	if value == "" {
		return styledRow(label, dimStyle.Render("unknown"))
	}
	return styledRow(label, valueStyle.Render(value))
}

// styledRow is row for a value that has already been styled.
func styledRow(label, rendered string) string {
	return labelStyle.Render(label) + rendered
}

func indent(s string, n int) string {
	pad := strings.Repeat(" ", n)
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = pad + l
	}
	return strings.Join(lines, "\n")
}
