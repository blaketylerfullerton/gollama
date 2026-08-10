package tui

import (
	"fmt"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// The second screen: pick what to run.
//
// It exists because the interesting constraint on a local model is memory, and
// memory cost isn't on a model card in a form you can act on. A card says
// "0.6B"; what you need to know is that this loader widens bf16 to float32, so
// those parameters land as 2.2GB of resident float, plus 229KB for every token
// of context, against whatever your machine has free right now. That arithmetic
// is three lines of code and nobody does it in their head, so the screen does it
// live for whatever the cursor is on.

// Outcome is how the picker ended.
type Outcome int

const (
	// Cancelled means quit — ctrl+c or q.
	Cancelled Outcome = iota
	// Selected means enter on a runnable model.
	Selected
	// Back means return to the welcome screen.
	Back
)

// contextEstimate is the context length the kv cache is sized at for display.
// It's a working length rather than the model's maximum: Qwen3's 40960-token
// window would cost 9.4GB of cache on the 0.6B, which is true but tells you
// nothing about the run you're about to start.
const contextEstimate = 4096

// minVisibleRows is the shortest the list is allowed to get. A directory full
// of checkpoints must not push the panels off the bottom of the terminal, but
// windowed down past this the list is more scrolling than list.
const minVisibleRows = 8

// listFrame is every row around the list that isn't the description/memory
// panel underneath it: the title line and its blank line, the list panel's own
// heading and border, and the gap before the panel below. The description and
// memory panel's own height is measured rather than guessed at, so a wider
// terminal that wraps their prose onto fewer lines gives the freed rows to the
// list instead of leaving them as padding inside a panel that didn't need them.
const listFrame = 8

// Picker is the bubbletea model for the model-selection screen.
type Picker struct {
	sys     sysinfo.Info
	models  []Model
	cursor  int
	top     int // first visible row, for scrolling
	outcome Outcome
	warn    string // set when enter lands on something that can't run
	w, h    int
}

var _ tea.Model = (*Picker)(nil)

// NewPicker builds the list from what's in root and puts the cursor on the
// first model that can actually run — there's no point opening on a row whose
// weights aren't downloaded.
func NewPicker(root string, sys sysinfo.Info) *Picker {
	p := &Picker{
		sys:    sys,
		models: Catalog(root),
		w:      100, h: 32,
	}
	for i, m := range p.models {
		if m.Installed {
			p.cursor = i
			break
		}
	}
	p.scroll()
	return p
}

// Outcome reports how the screen ended. Valid once the program has returned.
func (p *Picker) Outcome() Outcome { return p.outcome }

// Selection is the model under the cursor when the screen ended.
func (p *Picker) Selection() Model { return p.models[p.cursor] }

func (p *Picker) Init() tea.Cmd { return nil }

func (p *Picker) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		p.w, p.h = msg.Width, msg.Height
		// A resize changes how many rows the list has, so the window it was
		// scrolled to may no longer contain the cursor.
		p.scroll()
	case tea.KeyMsg:
		switch msg.String() {
		case "up", "k":
			p.move(-1)
		case "down", "j":
			p.move(1)
		case "home", "g":
			p.cursor, p.warn = 0, ""
			p.scroll()
		case "end", "G":
			p.cursor, p.warn = len(p.models)-1, ""
			p.scroll()
		case "enter", " ":
			return p, p.choose()
		case "b", "backspace", "left", "esc":
			p.outcome = Back
			return p, tea.Quit
		case "q", "ctrl+c":
			p.outcome = Cancelled
			return p, tea.Quit
		}
	}
	return p, nil
}

// move steps the cursor without wrapping. Wrapping in a list this short means
// pressing down one too many times silently jumps you to the top.
func (p *Picker) move(d int) {
	next := p.cursor + d
	if next < 0 || next >= len(p.models) {
		return
	}
	p.cursor, p.warn = next, ""
	p.scroll()
}

// visibleRows is how many models the list shows at the current height. A tall
// terminal gets the whole catalog without scrolling, which is the point of
// filling the screen rather than drawing a fixed-size card in the corner of it.
//
// The description/memory panel is rendered here at its natural height — not
// stretched to fill leftover space the way it used to be — so whatever room
// that panel doesn't need goes to the list instead of sitting as blank padding
// inside a box that already said everything it had to say.
func (p *Picker) visibleRows() int {
	inner := max(p.w-2*screenMargin, 24)
	mem := memPanelStyle.Render(p.memory())
	bottom := p.bottom(inner, 0, mem)
	chrome := listFrame + lipgloss.Height(bottom) + lipgloss.Height(p.bar())
	return min(max(p.h-chrome, minVisibleRows), max(len(p.models), 1))
}

// scroll keeps the cursor inside the visible window.
func (p *Picker) scroll() {
	rows := p.visibleRows()
	switch {
	case p.cursor < p.top:
		p.top = p.cursor
	case p.cursor >= p.top+rows:
		p.top = p.cursor - rows + 1
	}
	// A window that just grew can leave empty rows at the bottom of the list
	// while models are still scrolled off the top. Pull it back down onto them.
	p.top = max(0, min(p.top, len(p.models)-rows))
}

// choose accepts the highlighted model, or explains why it can't. A row whose
// weights aren't there is left on the list rather than hidden — knowing the
// model exists and what it would cost is most of the point — so enter on it has
// to say something rather than do nothing.
func (p *Picker) choose() tea.Cmd {
	m := p.Selection()
	if !m.Installed && !m.Demo {
		p.warn = m.Name + " isn't downloaded yet — run the command above, then come back"
		return nil
	}
	p.outcome = Selected
	return tea.Quit
}

func (p *Picker) View() string {
	// One cell of margin on each side, matching the welcome screen; two rows
	// off the top for the title and the blank line under it.
	bar := p.bar()
	inner := max(p.w-2*screenMargin, 24)

	// The list is the one panel that's every model at once rather than a
	// reading about the one under the cursor, so it's the one that gets the
	// full width — a table of numbers is exactly as wide as its widest row
	// wherever that lands, and stopping short of the terminal edge for no
	// reason reads as an accident. visibleRows already gave it whatever room
	// the description/memory panel below didn't need, so it's sized here at
	// its natural height rather than stretched further.
	list := panelStyle.Width(inner - 2).Render(p.list(inner - 6)) // less border + padding

	// Below it, the model's own summary beside what running it costs on this
	// machine — the memory column is a fixed width of number columns, so the
	// description takes whatever's left rather than the two splitting the row
	// down the middle. Rendered at its natural height: any leftover space in a
	// tall terminal went to the list above, not into padding here.
	mem := memPanelStyle.Render(p.memory())
	bottom := p.bottom(inner, 0, mem)

	// Same shape as the welcome screen: the title sits directly on top of the
	// block it names rather than off in the corner of the terminal.
	body := lipgloss.JoinVertical(lipgloss.Left,
		p.headerRow(lipgloss.Width(list)), "", list, bottom)
	return screen(p.w, p.h, body, bar)
}

// headerRow is the title with this machine's vitals pushed to the far end of
// the same line — the numbers that decide what fits are worth seeing before
// scanning a single row of the list, and a line that already exists is a
// cheaper place for them than a panel of their own.
func (p *Picker) headerRow(width int) string {
	title := header("GoLlama", "choose a model to start with", width)
	specs := dimStyle.Render(fmt.Sprintf("%s · %s ram · %s free",
		nameOr(p.sys.Host, "unknown"), sizeOr(p.sys.MemoryBytes), sizeOr(p.sys.AvailableBytes)))
	if gap := width - lipgloss.Width(title) - lipgloss.Width(specs); gap >= 2 {
		return title + strings.Repeat(" ", gap) + specs
	}
	return title
}

func nameOr(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func sizeOr(n uint64) string {
	if n == 0 {
		return "unknown"
	}
	return sysinfo.Bytes(int64(n))
}

// bottom lays the description beside the memory column, each stretched to the
// taller of the two so neither panel's border stops short of the other's. Below
// the width the memory column needs beside it, they stack instead — same
// threshold the welcome screen uses for the llama.
//
// When there isn't room for rows lines of prose, paragraphs come off the
// description one at a time until what's left fits — see about.
func (p *Picker) bottom(width, rows int, mem string) string {
	descWidth := width - lipgloss.Width(mem) - 1
	stacked := descWidth < listInnerWidth+panelChrome
	if stacked {
		descWidth = width
	}
	style := panelStyle.Width(descWidth - 2)

	descRoom := rows
	if !stacked {
		descRoom = max(rows, lipgloss.Height(mem))
	}

	notes := len(p.Selection().Notes)
	desc := stretch(style, descRoom, p.about(notes))
	for notes > 0 && lipgloss.Height(desc) > descRoom {
		notes--
		desc = stretch(style, descRoom, p.about(notes))
	}

	if stacked {
		return lipgloss.JoinVertical(lipgloss.Left, desc, mem)
	}
	if lipgloss.Height(desc) > lipgloss.Height(mem) {
		mem = memPanelStyle.Height(lipgloss.Height(desc) - 2).Render(p.memory())
	}
	return lipgloss.JoinHorizontal(lipgloss.Top, desc, " ", mem)
}

// --- the list ----------------------------------------------------------------

// fitColWidth fits the longest verdict — "recommended" — plus nothing extra;
// the column is right-aligned against the panel's own border so it needs no
// padding of its own.
const fitColWidth = 11

// listRowChrome is every cell a row spends on something other than the name:
// the "▸ " cursor, the four right-hand columns, and the gap in front of each
// one. The name column takes whatever width is left, which is what lets the
// row spread across the full panel instead of stopping at a fixed column.
//
// Has to match row's format string exactly — one space before the first
// column, two before the rest — or the row runs a cell past the panel it's
// rendered inside.
const listRowChrome = 2 /* prefix */ + 1 + 6 /* params */ + 2 + 8 /* size */ +
	2 + 8 /* status */ + 2 + fitColWidth

// list is the left panel: one row per model, with the numbers that decide
// between them, plus a verdict on whether it'll fit on this machine.
func (p *Picker) list(width int) string {
	rows := []string{heading("models"), ""}

	rec := p.recommended()
	end := min(p.top+p.visibleRows(), len(p.models))
	for i := p.top; i < end; i++ {
		rows = append(rows, p.row(i, width, i == rec))
	}
	if hidden := len(p.models) - end; hidden > 0 {
		rows = append(rows, dimStyle.Render(fmt.Sprintf("  %d more below", hidden)))
	}
	return strings.Join(rows, "\n")
}

// row renders one model, its name stretched to fill whatever room the fixed
// columns leave so the row reaches the far edge of the panel whatever the
// terminal width — the point of a table is that everything lines up, and a
// column of numbers huddled against the left border while the rest of the box
// sits empty doesn't read as one.
func (p *Picker) row(i, width int, recommended bool) string {
	m := p.models[i]

	size := sysinfo.Bytes(m.Arch.DiskBytes())
	if m.Installed {
		size = sysinfo.Bytes(m.OnDisk)
	}
	if m.Demo {
		size = "—"
	}

	nameWidth := max(width-listRowChrome, 8)
	text := fmt.Sprintf("%-*s %s  %s  %s  %s",
		nameWidth, trunc(m.Name, nameWidth),
		padLeft(params(m.Arch.Params()), 6),
		padLeft(size, 8),
		padLeft(status(m), 8),
		p.fit(m, recommended))

	if i == p.cursor {
		return selectedStyle.Render("▸ " + text)
	}
	return "  " + dimStyle.Render(text)
}

// status is the one word that says whether pressing enter will do anything.
func status(m Model) string {
	switch {
	case m.Demo:
		return "built in"
	case m.Installed:
		return "ready"
	default:
		return "get it"
	}
}

// recommended picks the biggest model that still fits this machine with room
// to spare, so the list can point at the best one rather than just the safest
// one. The demo model is never it — it always fits trivially, and pointing at
// the random model as the recommendation defeats the point of the column.
//
// It returns -1 when nothing fits, or when the machine didn't report enough
// memory to judge by.
func (p *Picker) recommended() int {
	if p.sys.Headroom() <= 0 {
		return -1
	}
	best := -1
	for i, m := range p.models {
		if m.Demo || p.residentFrac(m) >= 0.5 {
			continue
		}
		if best == -1 || m.Arch.Params() > p.models[best].Arch.Params() {
			best = i
		}
	}
	return best
}

// residentFrac is what running m would cost against what's free right now, the
// same fraction the gauge colours by.
func (p *Picker) residentFrac(m Model) float64 {
	head := p.sys.Headroom()
	if head <= 0 {
		return 0
	}
	ctx := min(contextEstimate, m.Arch.Context)
	resident := m.Arch.ResidentBytes() + m.Arch.KVBytes(ctx)
	return float64(resident) / float64(head)
}

// fit is the verdict in the rightmost column: this repo's actual recommendation
// for the biggest model worth running, or a plain fits/too large for everything
// else. It's the same headroom arithmetic as the memory panel and the gauge,
// collapsed to the one word this column has room for.
func (p *Picker) fit(m Model, recommended bool) string {
	if recommended {
		return keyStyle.Bold(true).Render(padLeft("recommended", fitColWidth))
	}
	if p.sys.Headroom() <= 0 {
		return dimStyle.Render(padLeft("—", fitColWidth))
	}
	if p.residentFrac(m) >= 1 {
		return warnStyle.Render(padLeft("too large", fitColWidth))
	}
	return dimStyle.Render(padLeft("fits", fitColWidth))
}

// --- the memory column -------------------------------------------------------

// memory is the right panel: this machine, then what the highlighted model does
// to it. Every line is derived from the row under the cursor, so moving the
// cursor is the whole interaction.
func (p *Picker) memory() string {
	m := p.Selection()
	a := m.Arch

	weights := a.ResidentBytes()
	ctx := min(contextEstimate, a.Context)
	kv := a.KVBytes(ctx)
	resident := weights + kv

	rows := []string{
		heading("MEMORY AFTER LOAD"),
		"",
	}

	if margin := p.sys.Headroom() / 2; margin > 0 {
		rows = append(rows,
			dimStyle.Render(fmt.Sprintf("recommendations assume you keep %s free", sysinfo.Bytes(int64(margin)))),
			"")
	}

	if !m.Installed && !m.Demo {
		rows = append(rows,
			memRow("to download", memValue(a.DiskBytes())),
			dimStyle.Render("  bf16, as HF ships it"),
			"")
	}

	rows = append(rows,
		memRow("resident", valueStyle.Render(padLeft(sysinfo.Bytes(resident), 10))),
	)

	// Free-after only means something when we know what was free to begin with.
	// When the answer is negative, the overshoot is the useful number — "0 B
	// free" is true of a model that misses by 200MB and of one that misses by
	// 20GB, and those are different decisions.
	if head := p.sys.Headroom(); head > 0 {
		if left := int64(head) - resident; left >= 0 {
			rows = append(rows, memRow("free after", valueStyle.Render(padLeft(sysinfo.Bytes(left), 10))))
		} else {
			rows = append(rows, memRow("over by", warnStyle.Render(padLeft(sysinfo.Bytes(-left), 10))))
		}
	}

	rows = append(rows, "", p.gauge(resident))
	return strings.Join(rows, "\n")
}

// gauge draws resident memory against what's free, with the verdict spelled out
// underneath. The bar is there to make "3.1 of 8.4" land faster than the digits
// do; the sentence is there because the digits are what you'd actually quote.
//
// The colour is the fraction, straight off the ramp — a bar that's a quarter
// full is also a quarter as bright, and one that's about to swap is the
// brightest thing on the screen. This used to be a green/yellow/orange verdict
// palette, which said the same thing in a way you had to have been taught. The
// ramp needs no legend: something getting brighter as it fills is what filling
// looks like.
func (p *Picker) gauge(resident int64) string {
	head := int64(p.sys.Headroom())
	if head <= 0 {
		return dimStyle.Render("this machine didn't report\nits memory, so there's\nnothing to compare against")
	}

	frac := float64(resident) / float64(head)
	verdict := "fits with room to spare"
	switch {
	case frac >= 1:
		verdict = "won't fit — will swap"
	case frac >= 0.85:
		verdict = "tight — expect pressure"
	case frac >= 0.5:
		verdict = "fits — most of what's free"
	}
	style := lipgloss.NewStyle().Foreground(amber.Of(frac))

	const width = 22
	filled := min(int(frac*width+0.5), width)
	// The trough sits a step below anything the fill can reach, so an almost
	// empty bar still reads as a bar with nothing in it rather than as a full
	// one that's been dimmed.
	bar := style.Render(strings.Repeat("█", filled)) +
		amber.Fg(amber.Ash).Render(strings.Repeat("░", width-filled))

	return bar + "\n" + style.Render(verdict)
}

func memRow(label, rendered string) string {
	return memLabelStyle.Render(label) + rendered
}

// memValue right-aligns a byte count into the number column, or says so when
// the platform didn't report one.
func memValue(n int64) string {
	if n <= 0 {
		return dimStyle.Render(padLeft("unknown", 10))
	}
	return valueStyle.Render(padLeft(sysinfo.Bytes(n), 10))
}

// --- the description ---------------------------------------------------------

// about is the full-width panel under the list: prose for the highlighted
// model, then the architecture it's prose about, then how to get it if it isn't
// here yet.
//
// notes caps how many paragraphs of prose are included. The prose is the first
// thing to give up on a terminal too short for all of this: the architecture
// line and the download command are what you'd act on, the paragraphs are what
// you'd read once.
func (p *Picker) about(notes int) string {
	m := p.Selection()
	a := m.Arch

	// The paragraphs go in unwrapped; the panel's Width does the wrapping, so
	// the same prose reflows instead of going ragged on a narrow terminal.
	rows := []string{heading(m.Name), ""}
	for _, para := range m.Notes[:min(notes, len(m.Notes))] {
		rows = append(rows, dimStyle.Render(para), "")
	}

	tied := "separate lm head"
	if a.TieEmbed {
		tied = "tied embeddings"
	}
	rows = append(rows,
		valueStyle.Render(fmt.Sprintf(
			"%d layers · %d q heads over %d kv heads × %d dims · %d-wide residual stream",
			a.NLayer, a.NHead, a.NKVHead, a.HeadDim, a.NEmbed)),
		valueStyle.Render(fmt.Sprintf(
			"%s parameters · %d vocabulary · %d max context · %s",
			params(a.Params()), a.VocabSize, a.Context, tied)))

	if cmd := m.Download(); cmd != nil && !m.Installed {
		rows = append(rows, "")
		for _, line := range cmd {
			rows = append(rows, keyStyle.Render(line))
		}
	}
	return strings.Join(rows, "\n")
}

// bar is the toolbar along the bottom.
//
// A complaint about the row under the cursor goes on a second line inside it
// rather than replacing the keys, which is what it used to do: the keys are the
// frame, and a screen that answers a keystroke by taking the keys away is a
// worse answer than one that explains itself underneath them.
func (p *Picker) bar() string {
	keys := []string{
		keyStyle.Render("↑↓") + dimStyle.Render(" choose"),
		keyStyle.Render("enter") + dimStyle.Render(" load it"),
		keyStyle.Render("b") + dimStyle.Render(" back"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	left := strings.Join(keys, dimStyle.Render(" · "))
	if p.warn != "" {
		left += "\n" + warnStyle.Render(p.warn)
	}
	return toolbar(p.w, left, "")
}

// trunc clips a name to fit its column, with an ellipsis so it's obvious that's
// what happened.
func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

// ShowPicker runs the model-selection screen on its own alternate screen.
func ShowPicker(root string, sys sysinfo.Info) (*Picker, error) {
	p := NewPicker(root, sys)
	if _, err := tea.NewProgram(p, tea.WithAltScreen()).Run(); err != nil {
		return p, err
	}
	return p, nil
}
