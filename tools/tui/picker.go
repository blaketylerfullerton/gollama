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

// maxVisibleRows caps the list so a directory full of checkpoints can't push
// the panels off the bottom of the terminal.
const maxVisibleRows = 8

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

// scroll keeps the cursor inside the visible window.
func (p *Picker) scroll() {
	switch {
	case p.cursor < p.top:
		p.top = p.cursor
	case p.cursor >= p.top+maxVisibleRows:
		p.top = p.cursor - maxVisibleRows + 1
	}
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
	// One cell of margin on each side, matching the welcome screen.
	inner := max(p.w-2, 24)

	// The memory column is always the taller of the two, so the list is padded
	// out to match it. Without that the border under the list stops halfway up
	// the panel beside it, which reads as a rendering bug rather than as a
	// short list.
	mem := memPanelStyle.Render(p.memory())
	list := listPanelStyle.Height(lipgloss.Height(mem) - 2).Render(p.list())

	top := lipgloss.JoinHorizontal(lipgloss.Top, list, " ", mem)
	if topWidth > inner {
		top = lipgloss.JoinVertical(lipgloss.Left, listPanelStyle.Render(p.list()), mem)
	}

	// The description runs the full width of whatever's above it, so the three
	// panels read as one block rather than as a wide box under two narrow ones.
	about := panelStyle.Width(min(lipgloss.Width(top), inner) - 2).Render(p.about())

	return strings.Join([]string{
		"",
		" " + titleStyle.Render("GoLlama") + " " +
			subtitleStyle.Render("choose a model to start with"),
		"",
		indent(top, 1),
		indent(about, 1),
		"",
		" " + p.footer(),
		"",
	}, "\n")
}

// --- the list ----------------------------------------------------------------

// list is the left panel: one row per model, with the two numbers that decide
// between them — how many parameters, and how much disk that is.
func (p *Picker) list() string {
	rows := []string{heading("models"), ""}

	end := min(p.top+maxVisibleRows, len(p.models))
	for i := p.top; i < end; i++ {
		rows = append(rows, p.row(i))
	}
	if hidden := len(p.models) - end; hidden > 0 {
		rows = append(rows, dimStyle.Render(fmt.Sprintf("  %d more below", hidden)))
	}
	return strings.Join(rows, "\n")
}

// row renders one model. The columns are fixed width so the sizes line up into
// something you can scan down, which is the only reason to show them at all.
func (p *Picker) row(i int) string {
	m := p.models[i]

	size := sysinfo.Bytes(m.Arch.DiskBytes())
	if m.Installed {
		size = sysinfo.Bytes(m.OnDisk)
	}
	if m.Demo {
		size = "—"
	}

	text := fmt.Sprintf("%-17s %s  %s  %s",
		trunc(m.Name, 17),
		padLeft(params(m.Arch.Params()), 6),
		padLeft(size, 8),
		padLeft(status(m), 8))

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
		heading("memory after load"),
		"",
		memRow("total ram", memValue(int64(p.sys.MemoryBytes))),
		memRow("available", memValue(int64(p.sys.AvailableBytes))),
		dimStyle.Render("  an estimate, not a promise"),
		"",
	}

	if !m.Installed && !m.Demo {
		rows = append(rows,
			memRow("to download", memValue(a.DiskBytes())),
			dimStyle.Render("  bf16, as HF ships it"),
			"")
	}

	rows = append(rows,
		memRow("weights", memValue(weights)),
		dimStyle.Render("  bf16 on disk → f32 in ram"),
		memRow("kv / token", memValue(a.KVBytesPerToken())),
		memRow(fmt.Sprintf("kv @ %d", ctx), memValue(kv)),
		dimStyle.Render(strings.Repeat("─", memInnerWidth)),
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
func (p *Picker) about() string {
	m := p.Selection()
	a := m.Arch

	// The paragraphs go in unwrapped; the panel's Width does the wrapping, so
	// the same prose reflows instead of going ragged on a narrow terminal.
	rows := []string{heading(m.Name), ""}
	for _, para := range m.Notes {
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

func (p *Picker) footer() string {
	if p.warn != "" {
		return warnStyle.Render(p.warn)
	}
	keys := []string{
		keyStyle.Render("↑↓") + dimStyle.Render(" choose"),
		keyStyle.Render("enter") + dimStyle.Render(" load it"),
		keyStyle.Render("b") + dimStyle.Render(" back"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	return strings.Join(keys, dimStyle.Render(" · "))
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
