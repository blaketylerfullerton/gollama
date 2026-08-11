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

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
	"github.com/blaketylerfullerton/GoLlama/tools/history"
	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// Choice is what the user did with the welcome screen.
type Choice int

const (
	// Quit means they backed out — ctrl+c, q, or esc. The caller should exit
	// without loading anything.
	Quit Choice = iota
	// Run means they picked "select a model" — go to the picker.
	Run
	// ShowAbout means they picked "what is GoLlama" — go read the about page,
	// then come back to this menu.
	ShowAbout
	// ShowHistory means they picked "past conversations" — go read saved
	// chats, then come back to this menu.
	ShowHistory
)

// menuItem is one box of the start menu: what it's called, what picking it
// leads to, and the detail panel shown while it's highlighted.
type menuItem struct {
	title  string
	choice Choice
	// detail renders the content panel for this item. It's a func rather than
	// a precomputed string because the machine specs need the terminal width
	// to wrap the download command, and that isn't known until View runs.
	detail func(w *Welcome) string
}

var menuItems = []menuItem{
	{title: "select a model", choice: Run, detail: (*Welcome).weightsBlurb},
	{title: "what is GoLlama", choice: ShowAbout, detail: (*Welcome).aboutBlurb},
	{title: "past conversations", choice: ShowHistory, detail: (*Welcome).historyBlurb},
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
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
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
	sys          sysinfo.Info
	ckpt         Checkpoint
	historyCount int // how many saved conversations to mention in the third box
	choice       Choice
	cursor       int // which menuItems row is highlighted
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
		ckpt:         ScanCheckpoint(checkpointDir),
		historyCount: history.Count(),
		w:            100, h: 32,
	}
}

// Choice reports what the user picked. Valid once the program has returned.
func (w *Welcome) Choice() Choice { return w.choice }

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
		case "left", "h":
			w.cursor = max(w.cursor-1, 0)
		case "right", "l":
			w.cursor = min(w.cursor+1, len(menuItems)-1)
		case "enter", " ":
			w.choice = menuItems[w.cursor].choice
			return w, tea.Quit
		case "q", "esc", "ctrl+c":
			w.choice = Quit
			return w, tea.Quit
		}
	}
	return w, nil
}

// minSpecsWidth is the narrowest the panel can be rendered and still hold its
// longest line — the huggingface-cli invocation, which is the one thing on the
// screen that has to be copied verbatim.
const minSpecsWidth = 48

const subtitle = "a transformer you can read, one token at a time"

// menuChrome is a box's padding (4) plus border (2), the same convention
// panelChrome uses elsewhere — see style.go.
const menuChrome = 6

// menuMinWidth is the narrowest the menu can be drawn without wrapping a
// title onto a second line: each box wide enough for its own title plus
// chrome, side by side with a gap between each pair.
func menuMinWidth() int {
	const gap = 2
	longest := 0
	for _, item := range menuItems {
		longest = max(longest, len(item.title)+2) // +2 for the "▸ " cursor
	}
	n := len(menuItems)
	return n*(longest+menuChrome) + (n-1)*gap
}

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

// menu is the side-by-side option boxes underneath the machine panel: what you
// can do from here, toggled left/right, with the highlighted box picked out
// the same way the picker's cursor points at the memory column.
//
// Below menuMinWidth there isn't room for every title on one line without
// wrapping, so it falls back to a single-line title strip over one detail
// panel — the same shape the two-item menu used before there were three of
// them to fit, kept for whatever terminal is too narrow for the boxes.
func (w *Welcome) menu(width, rows int) string {
	if width < menuMinWidth() {
		return w.menuStacked(width, rows)
	}
	// n boxes with a gap between each pair, splitting whatever's left evenly.
	const gap = 2
	n := len(menuItems)
	boxWidth := (width - gap*(n-1)) / n
	boxes := make([]string, n)
	for i, item := range menuItems {
		style := panelStyle.Width(boxWidth - 2)
		title := "  " + dimStyle.Render(item.title)
		if i == w.cursor {
			style = style.BorderForeground(amber.At(amber.Hot))
			title = selectedStyle.Render("▸ " + item.title)
		}
		body := lipgloss.JoinVertical(lipgloss.Left, title, "", item.detail(w))
		boxes[i] = stretch(style, rows, body)
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, joinWithGaps(boxes, gap)...)
}

// menuStacked is the narrow-terminal fallback: one title per line rather than
// a row of them, since a terminal too narrow for the boxes side by side is
// also too narrow for their titles side by side in a strip. The highlighted
// one's detail goes underneath in a single panel that takes the full width
// instead of a third of it.
func (w *Welcome) menuStacked(width, rows int) string {
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

// joinWithGaps interleaves a fixed-width blank column between each element,
// for lipgloss.JoinHorizontal calls over a slice whose length isn't fixed at
// two.
func joinWithGaps(boxes []string, gap int) []string {
	out := make([]string, 0, 2*len(boxes)-1)
	for i, b := range boxes {
		if i > 0 {
			out = append(out, strings.Repeat(" ", gap))
		}
		out = append(out, b)
	}
	return out
}

// machineSpecs is the always-visible panel above the menu: what this machine
// is, so the two options below it can be judged against real hardware rather
// than in the abstract.
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
	)
	return strings.Join(rows, "\n")
}

// weightsBlurb is the detail panel for the menu's first box: what's on disk to
// run on this machine.
func (w *Welcome) weightsBlurb() string {
	rows := append([]string{heading("Weights")}, w.checkpointRows()...)
	return strings.Join(rows, "\n")
}

func (w *Welcome) checkpointRows() []string {
	if !w.ckpt.Present {
		return []string{
			styledRow("model", warnStyle.Render("none found")),
			row("expected", w.ckpt.Dir),
			"",
			dimStyle.Render("  huggingface-cli download Qwen/Qwen3-0.6B"),
			dimStyle.Render("    --local-dir " + w.ckpt.Dir),
			"",
			dimStyle.Render("Without it GoLlama runs a tiny random model, so"),
			dimStyle.Render("every stage still works and the numbers are noise."),
		}
	}
	return []string{
		row("model", filepath.Base(w.ckpt.Dir)),
		row("on disk", sysinfo.Bytes(w.ckpt.Bytes)),
		// The loader widens every stored bf16 to a float32, so what ends up in
		// memory is twice what's on disk. Worth saying out loud: it's the
		// difference between "fits easily" and "swapping" on a small machine.
		styledRow("resident", valueStyle.Render(sysinfo.Bytes(w.ckpt.Bytes*2))+
			dimStyle.Render("  (bf16 → f32)")),
		"",
		dimStyle.Render("Every matmul is scalar Go on the CPU, so expect"),
		dimStyle.Render("tokens in the hundreds of milliseconds, not micro-"),
		dimStyle.Render("seconds. That slowness is the point: it's readable."),
	}
}

// aboutBlurb is the detail panel for the menu's second row: enough to say what
// pressing enter leads to, not the whole page — that's what the about page
// itself is for.
func (w *Welcome) aboutBlurb() string {
	rows := []string{
		heading("What is GoLlama"),
		"",
		dimStyle.Render(aboutNotes[0]),
		"",
		valueStyle.Render("enter") + dimStyle.Render(" to read the rest — architecture, how the"),
		dimStyle.Render("inspector works, and what's slow on purpose and why."),
	}
	return strings.Join(rows, "\n")
}

// historyBlurb is the detail panel for the menu's third box: how many
// conversations there are to read back, not their contents — that's what the
// history page itself is for.
func (w *Welcome) historyBlurb() string {
	if w.historyCount == 0 {
		return strings.Join([]string{
			heading("Past conversations"),
			"",
			dimStyle.Render("Nothing saved yet. Every conversation on the chat screen is written"),
			dimStyle.Render("here as it happens, so it's still here next time you open GoLlama."),
		}, "\n")
	}
	plural := "s"
	if w.historyCount == 1 {
		plural = ""
	}
	return strings.Join([]string{
		heading("Past conversations"),
		"",
		fmt.Sprintf("%s saved conversation%s", valueStyle.Render(fmt.Sprint(w.historyCount)), plural),
		"",
		valueStyle.Render("enter") + dimStyle.Render(" to read back what was said, without"),
		dimStyle.Render("reloading the model or the engine behind it."),
	}, "\n")
}

// bar is the toolbar along the bottom: the keys this screen answers to on the
// left, and the other way into the program on the right.
func (w *Welcome) bar() string {
	keys := []string{
		keyStyle.Render("←→") + dimStyle.Render(" choose"),
		keyStyle.Render("enter") + dimStyle.Render(" select"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	return toolbar(w.w, strings.Join(keys, dimStyle.Render(" · ")),
		dimStyle.Render("go run ./cmd/inspect to step through a pass"))
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
