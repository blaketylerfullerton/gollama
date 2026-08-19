// Package tui is the terminal front end for GoLlama.
//
// The first thing it shows is a welcome screen: a list of tools on the left,
// the highlighted one's detail on the right, and the machine it's about to
// run on along the bottom. That last part is the point — a 0.6B model in
// scalar Go is entirely at the mercy of the cores and the memory listed
// there, so seeing them before the first token makes the speed that follows
// legible rather than disappointing.
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
	// ToolWatermark runs the prompt twice — plain, and through a
	// SynthID-Text-style tournament sampler — and shows what a detector
	// reads off each. Not another lens onto this model's internals like the
	// four above it; it's a demo of a technique applied on top of sampling,
	// which is why it opens its own screen (see root.go's openWatermark)
	// rather than another Inspect view.
	ToolWatermark
	// ToolChat talks to the model turn by turn, same as every other tool a
	// peer of in this list rather than the separate program it used to be.
	ToolChat
	// ToolModel opens the picker on its own, with no analysis view behind
	// it — for browsing or downloading a checkpoint without committing to
	// a tool to run it through. See root.go's advance, which sends it
	// straight back to the welcome menu instead of on to chat or inspect.
	ToolModel
)

// menuItem is one row of the welcome menu: what it's called, which tool
// picking it opens, and the detail panel shown while it's highlighted.
type menuItem struct {
	title string
	tool  Tool
	// run says whether enter does anything on this row. Every row but
	// Machine leads somewhere; Machine is there to be highlighted for its
	// detail panel and nothing else, since the bottom bar already says what
	// it would say.
	run bool
	// detail renders the content panel for this item. It's a func rather than
	// a precomputed string because some detail panels need the terminal width
	// to wrap, and that isn't known until View runs.
	detail func(w *Welcome) string
}

// toolItems are the trace-viewer tools, top to bottom in the order about.go's
// third paragraph describes them.
var toolItems = []menuItem{
	{title: "Attention", tool: ToolAttention, run: true, detail: (*Welcome).attentionBlurb},
	{title: "Logit Lens", tool: ToolLens, run: true, detail: (*Welcome).lensBlurb},
	{title: "Attribution", tool: ToolAttribution, run: true, detail: (*Welcome).attributionBlurb},
	{title: "Ablation", tool: ToolAblation, run: true, detail: (*Welcome).ablationBlurb},
	{title: "Watermark", tool: ToolWatermark, run: true, detail: (*Welcome).watermarkBlurb},
}

// otherItems is everything on the menu that isn't a trace tool: talk to the
// model directly, manage which checkpoint is on disk, or just read what's
// under it.
var otherItems = []menuItem{
	{title: "Chat", tool: ToolChat, run: true, detail: (*Welcome).chatBlurb},
	{title: "Model", tool: ToolModel, run: true, detail: (*Welcome).modelBlurb},
	{title: "Machine", detail: (*Welcome).machineBlurb},
}

// menuItems is toolItems and otherItems back to back, so one cursor can move
// through both sections without the rest of the screen caring where the
// section boundary falls.
var menuItems = append(append([]menuItem{}, toolItems...), otherItems...)

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

// Init names the terminal tab. Nothing on this screen animates any more.
func (w *Welcome) Init() tea.Cmd {
	return tea.SetWindowTitle("🦙 GoLlama")
}

func (w *Welcome) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		w.w, w.h = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			w.cursor = max(w.cursor-1, 0)
		case "down", "j":
			w.cursor = min(w.cursor+1, len(menuItems)-1)
		case "enter", " ":
			// Machine is highlightable but leads nowhere — the bottom bar
			// already says what its detail panel says, so enter on that row
			// is a no-op rather than a Run with nothing to run.
			if item := menuItems[w.cursor]; item.run {
				w.choice = Run
				w.tool = item.tool
				return w, done
			}
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

	head := hero("GoLlama", subtitle, inner)
	machine := w.machineRow(inner)
	main := w.main(inner, rows-lipgloss.Height(head)-lipgloss.Height(machine)-2)

	body := lipgloss.JoinVertical(lipgloss.Left, head, "", main, "", machine)
	return screen(w.w, w.h, body, bar)
}

// main is the selector on the left and the highlighted row's detail on the
// right, side by side when there's room for both and stacked — detail under
// list — when there isn't.
//
// listOuter is the selector panel's own rendered width, gap included in the
// split below it — see style.go's panel-arithmetic comments (also on
// about.go's layout): a panel meant to render at total width X is built with
// panelStyle.Width(X-2).
func (w *Welcome) main(width, rows int) string {
	const listOuter = 30
	if detailOuter := width - listOuter - 2; detailOuter >= minSpecsWidth/2 {
		list := stretch(panelStyle.Width(listOuter-2), rows, w.selectorList())
		detail := stretch(panelStyle.Width(detailOuter-2), rows, menuItems[w.cursor].detail(w))
		return lipgloss.JoinHorizontal(lipgloss.Top, list, "  ", detail)
	}
	list := stretch(panelStyle.Width(width-2), 0, w.selectorList())
	detail := stretch(panelStyle.Width(width-2), rows-lipgloss.Height(list)-1,
		menuItems[w.cursor].detail(w))
	return lipgloss.JoinVertical(lipgloss.Left, list, "", detail)
}

// selectorList is the two sections of the welcome menu, one cursor moving
// through both: "Lenses" — the trace tools — first, then "Other" — chat,
// the model picker, and the machine this is all running on.
func (w *Welcome) selectorList() string {
	rows := make([]string, 0, len(toolItems)+len(otherItems)+4)
	rows = append(rows, heading("Lenses"))
	idx := 0
	for _, item := range toolItems {
		rows = append(rows, w.menuRow(item, idx))
		idx++
	}
	rows = append(rows, "", heading("Other"))
	for _, item := range otherItems {
		rows = append(rows, w.menuRow(item, idx))
		idx++
	}
	return strings.Join(rows, "\n")
}

// menuRow renders one selector row, highlighted when idx is under the cursor.
func (w *Welcome) menuRow(item menuItem, idx int) string {
	if idx == w.cursor {
		return selectedStyle.Render("▸ " + item.title)
	}
	return "  " + dimStyle.Render(item.title)
}

// machineRow is the bottom strip: what this machine is and what's on disk to
// run on it, always visible regardless of which row is highlighted — one
// line rather than the panel this used to be, now that the selector and its
// detail box fill the middle of the screen instead.
func (w *Welcome) machineRow(width int) string {
	s := w.sys
	fields := []string{
		statField("host", s.Host),
		statField("chip", s.CPU),
		statField("cores", s.CoreSummary()),
	}
	if s.GPU != "" {
		fields = append(fields, statField("gpu", s.GPU))
	}
	// Free memory is shown beside the total because it's the one that decides
	// whether a checkpoint loads comfortably, and on a machine that's been up
	// for a week the two are nothing like each other.
	mem := s.Memory()
	if s.AvailableBytes > 0 {
		mem = fmt.Sprintf("%s (%s free)", s.Memory(), s.Available())
	}
	fields = append(fields,
		statField("memory", mem),
		statField("platform", s.Platform()),
		w.weightsField(),
	)
	line := strings.Join(fields, dimStyle.Render("   ·   "))
	return panelStyle.Width(width - 2).Render(line)
}

// statField is one label/value pair in the bottom machine row. An empty
// value reads as "unknown" rather than as a blank field, since on an
// unsupported platform half these fields legitimately come back empty.
func statField(label, value string) string {
	if value == "" {
		value = "unknown"
	}
	return dimStyle.Render(label+" ") + valueStyle.Render(value)
}

// weightsField is statField's counterpart for checkpoint status: what's on
// disk to run these tools against, so it's never lost regardless of which
// row is highlighted. The picker — one keypress away, on every tool —
// downloads a missing model itself; this just says whether it has to.
func (w *Welcome) weightsField() string {
	if !w.ckpt.Present {
		return dimStyle.Render("weights ") + warnStyle.Render("none found")
	}
	return dimStyle.Render("weights ") + valueStyle.Render(fmt.Sprintf("%s (%s)",
		filepath.Base(w.ckpt.Dir), sysinfo.Bytes(w.ckpt.Bytes)))
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
		"",
		dimStyle.Render("example — ablating layer 9, head 3:"),
		ablationExample(),
	}, "\n")
}

// watermarkBlurb is the detail panel for the menu's fifth row.
func (w *Welcome) watermarkBlurb() string {
	return strings.Join([]string{
		heading("Watermark"),
		"",
		dimStyle.Render("Generate the same prompt plain and through a SynthID-Text-style"),
		dimStyle.Render("tournament sampler, then run a detector over both — the detector"),
		dimStyle.Render("reads a statistical signature out of text, no metadata involved."),
		"",
		dimStyle.Render("example — the detector's read on each:"),
		watermarkExample(),
	}, "\n")
}

// attentionBlurb is the detail panel for the menu's second row.
func (w *Welcome) attentionBlurb() string {
	return strings.Join([]string{
		heading("Attention"),
		"",
		dimStyle.Render("See what each head attended to, layer by layer — the causal"),
		dimStyle.Render("weights over every earlier token, coloured by magnitude."),
		"",
		dimStyle.Render("example — one query token's weights over what came before:"),
		attentionExample(),
	}, "\n")
}

// attributionBlurb is the detail panel for the menu's third row.
func (w *Welcome) attributionBlurb() string {
	return strings.Join([]string{
		heading("Attribution"),
		"",
		dimStyle.Render("See which components — which head, which MLP — actually pushed"),
		dimStyle.Render("the final answer, and by how much, positive or negative."),
		"",
		dimStyle.Render(`example — what pushed "Paris":`),
		attributionExample(),
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
		"",
		dimStyle.Render(`example — predicting after "the capital of France is":`),
		lensExample(),
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
	rows = append(rows,
		"",
		dimStyle.Render("example turn:"),
		"  "+youStyle.Render("you")+"    "+valueStyle.Render("what's the capital of France?"),
		"  "+modelStyle.Render("model")+"  "+modelReplyStyle.Render("Paris."),
	)
	return strings.Join(rows, "\n")
}

// modelBlurb is the detail panel for the Other section's second row: the
// picker opened on its own, with no analysis tool waiting behind it.
func (w *Welcome) modelBlurb() string {
	rows := []string{
		heading("Model"),
		"",
		dimStyle.Render("Browse installed checkpoints or pull a new one from HuggingFace —"),
		dimStyle.Render("the same picker every tool above opens once it needs a model."),
		"",
		row("weights", w.weightsSummary()),
	}
	if w.ckpt.Present {
		rows = append(rows, w.residentEstimateRow())
	}
	return strings.Join(rows, "\n")
}

// residentEstimateRow guesses what the checkpoint on disk will cost once
// loaded, against what this machine actually has free. The 2x is the same
// one catalog.go's Arch.ResidentBytes documents: HuggingFace ships bf16, the
// loader widens every weight to float32 on load. It's an estimate rather
// than the picker's own exact figure because this screen only knows the
// bytes on disk, not the checkpoint's architecture.
func (w *Welcome) residentEstimateRow() string {
	resident := w.ckpt.Bytes * 2
	label := fmt.Sprintf("~%s resident", sysinfo.Bytes(resident))
	if w.sys.AvailableBytes <= 0 {
		return styledRow("estimate", dimStyle.Render(label))
	}
	frac := min(float64(resident)/float64(w.sys.AvailableBytes), 1)
	return styledRow("estimate", valueStyle.Render(label)+
		dimStyle.Render(fmt.Sprintf("  of %s free  ", w.sys.Available()))+bar(frac, 14))
}

// weightsSummary is weightsField's plain-text counterpart for modelBlurb,
// which renders its own row label via row() rather than the bottom bar's
// inline style.
func (w *Welcome) weightsSummary() string {
	if !w.ckpt.Present {
		return "none found"
	}
	return fmt.Sprintf("%s (%s on disk)", filepath.Base(w.ckpt.Dir), sysinfo.Bytes(w.ckpt.Bytes))
}

// machineBlurb is the detail panel for the Other section's third row: the
// full hardware picture, one field per line, for whoever wants more than the
// bottom bar's single line says. This row leads nowhere on enter — see
// otherItems — it exists to be highlighted.
func (w *Welcome) machineBlurb() string {
	s := w.sys
	rows := []string{
		heading("Machine"),
		"",
		row("host", s.Host),
		row("chip", s.CPU),
		row("cores", s.CoreSummary()),
	}
	if s.GPU != "" {
		rows = append(rows, row("gpu", s.GPU))
	}
	memory := row("memory", s.Memory())
	if s.AvailableBytes > 0 {
		memory = styledRow("memory", valueStyle.Render(s.Memory())+
			dimStyle.Render("  "+s.Available()+" free"))
	}
	rows = append(rows,
		memory,
		row("platform", s.Platform()),
		row("runtime", fmt.Sprintf("%s · GOMAXPROCS %d", s.GoVersion, s.GOMAXPROCS)),
		row("ckpt dir", w.dir),
		"",
		dimStyle.Render("Every matmul here is plain scalar Go — no BLAS, no GPU kernels —"),
		dimStyle.Render("so the numbers above are the whole budget a token has to work with."),
	)
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

// --- welcome-menu example previews -------------------------------------------
//
// Each tool's blurb ends with a miniature of what pressing enter on it opens,
// built from the same primitives the real inspect screens render with — bar
// and signedBar, from inspect_views.go — so this reads as an honest preview
// rather than a mockup with its own visual language. The numbers are fixed
// rather than sampled from an actual run: a real one needs a loaded model and
// a prompt, and neither exists yet on this screen.

// attentionExample previews one query token's weights over what came before —
// attentionView's own row, for a query rather than the whole matrix.
func attentionExample() string {
	toks := []string{"The", "capital", "of", "France", "is"}
	weights := []float64{0.02, 0.05, 0.11, 0.68, 0.14}
	var b strings.Builder
	for i, t := range toks {
		fmt.Fprintf(&b, "  %-9s %s %4.0f%%\n", t, bar(weights[i], 16), weights[i]*100)
	}
	return strings.TrimRight(b.String(), "\n")
}

// lensExample previews lensView's own layer/prediction/probability row, at
// three depths standing in for early, middle, and late in a real pass.
func lensExample() string {
	rows := []struct {
		layer, guess string
		prob         float64
	}{
		{"layer 0", `" the"`, 0.12},
		{"layer 12", `" Paris"`, 0.54},
		{"layer 23", `" Paris"`, 0.91},
	}
	var b strings.Builder
	for _, r := range rows {
		fmt.Fprintf(&b, "  %-9s%-10s %s %4.0f%%\n", r.layer, r.guess, bar(r.prob, 16), r.prob*100)
	}
	return strings.TrimRight(b.String(), "\n")
}

// attributionExample previews attributionView's signed push per component.
func attributionExample() string {
	rows := []struct {
		name string
		v    float64
	}{
		{"embed", 0.12},
		{"L4 head 3", 0.41},
		{"L9 MLP", -0.08},
	}
	var b strings.Builder
	for _, r := range rows {
		fmt.Fprintf(&b, "  %-10s %+.2f %s\n", r.name, r.v, signedBar(r.v, 0.41, 10))
	}
	return strings.TrimRight(b.String(), "\n")
}

// ablationExample previews ablationView's baseline-vs-ablated comparison.
func ablationExample() string {
	return strings.Join([]string{
		fmt.Sprintf("  %-9s %-14s %s", "baseline", `"Paris" 91%`, bar(0.91, 16)),
		fmt.Sprintf("  %-9s %-14s %s", "ablated", `"France" 38%`, bar(0.38, 16)),
		dimStyle.Render("  top prediction flips — this head was load-bearing here"),
	}, "\n")
}

// watermarkExample previews watermarkView's detector readout — same shape as
// ablationExample's baseline-vs-ablated comparison, since this is the same
// kind of two-run comparison, just over the detector's z-score instead of a
// top prediction.
func watermarkExample() string {
	return strings.Join([]string{
		fmt.Sprintf("  %-11s %-16s %s", "plain", "z  -0.5", bar(0.05, 16)),
		fmt.Sprintf("  %-11s %-16s %s", "watermarked", "z  4.9", bar(0.6, 16)),
		dimStyle.Render("  z above ~4 reads as watermarked — ordinary text has no reason to get there"),
	}, "\n")
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
