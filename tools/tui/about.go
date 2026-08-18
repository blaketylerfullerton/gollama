package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// The page behind the welcome menu's second item: what this program actually
// is, for whoever landed here without having read the README first.

// aboutNotes is the prose, one paragraph per entry so the panel can wrap each
// one to whatever width it's given rather than to whatever width looked right
// in the source file. Ordered the way the README itself argues its case: what
// this is, why it's slow on purpose, then where to go to watch it think.
var aboutNotes = []string{
	"GoLlama is a transformer — currently Qwen3 — rewritten by hand in Go to be read, not just run. It runs real pretrained Qwen3-0.6B weights, but the engine underneath has no third-party dependencies at all; only this terminal UI pulls anything in.",
	"Inference only, no training. Every matmul is plain scalar Go, which is slow on purpose — hundreds of milliseconds a token rather than microseconds — so a token landing is something you can watch happen rather than a number that appears.",
	"The point isn't speed, it's that every stage is inspectable. -v narrates a forward pass step by step: shapes, per-layer magnitudes, rotary tables, attention weights. The logit lens projects the residual stream through the output head at every layer, so you can watch an answer get found — early layers guess generic words, and the real answer usually only takes the lead a handful of layers before the end.",
	"Ablation, Attention, Attribution, and Logit Lens — the other four rows on the welcome menu — open an interactive trace viewer: type a prompt, run it, then step between generated tokens to see what each one attended to, what pushed its answer, and where that answer settled. Chat doesn't show this live, but every turn is recorded as it happens — the history key (h) on the welcome menu opens the same step-through over anything you've said to it.",
	"Underneath both viewers is one recorder — tools/trace — that turns a forward pass into a file instead of driving a UI directly, so the engine never learns an inspector exists. It's JSON Lines: one header, then one line per event, of five kinds:",
	"Grouped-query attention, rotary position embeddings, QK-norm, SwiGLU, RMSNorm, tied embeddings — Qwen3's architecture, matching Llama everywhere it can and diverging where it has to. None of it is optimized; all of it is meant to be legible.",
}

// traceEventDocs is what tools/trace's five Kind values actually record — kept
// as its own boxed panel rather than folded into the prose above it, since a
// reference list reads better set apart than run into a paragraph around it.
var traceEventDocs = []struct{ kind, desc string }{
	{"stage", "the residual stream's shape and magnitude at each point it's touched"},
	{"attention", "one head's causal weights over every earlier token"},
	{"rotary", "a head's query vector either side of the rotation, and how much it moved"},
	{"logit_lens", "what the whole model would have guessed if it stopped at that layer"},
	{"note", "commentary that isn't a tensor"},
}

// About is the bubbletea model for the "what is GoLlama" page.
type About struct {
	vp      viewport.Model
	outcome AboutOutcome
	w, h    int
}

var _ tea.Model = (*About)(nil)

// AboutOutcome is how the page ended.
type AboutOutcome int

const (
	// AboutBack means they backed out to the menu — b, esc, or backspace.
	AboutBack AboutOutcome = iota
	// AboutQuit means they left the program entirely.
	AboutQuit
)

// NewAbout builds the page.
func NewAbout() *About {
	return &About{vp: viewport.New(0, 0), w: 100, h: 32}
}

// Outcome reports how the page ended. Valid once the program has returned.
func (a *About) Outcome() AboutOutcome { return a.outcome }

func (a *About) Init() tea.Cmd { return nil }

func (a *About) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.w, a.h = msg.Width, msg.Height
		a.layout()
		return a, nil
	case tea.KeyMsg:
		switch msg.String() {
		case "b", "backspace", "esc":
			a.outcome = AboutBack
			return a, done
		case "q", "ctrl+c":
			a.outcome = AboutQuit
			return a, done
		default:
			var cmd tea.Cmd
			a.vp, cmd = a.vp.Update(msg)
			return a, cmd
		}
	}
	return a, nil
}

func (a *About) layout() {
	bar := a.bar()
	inner := max(a.w-2*screenMargin, minSpecsWidth)
	rows := bodyRows(a.h, bar) - 2 // less the title and the blank line under it

	// The panel renders at panelStyle.Width(inner-2), and that call's own
	// padding (4) is on top of the width it's given rather than inside it — so
	// the text column inside the border and padding is inner-2-4, not inner-4.
	// See picker.go's list/mem panels for the same arithmetic.
	a.vp.Width = inner - 6
	a.vp.Height = max(rows-2, 3)
	a.vp.SetContent(a.body())
}

// body wraps the prose to the viewport's width before handing it over.
//
// The viewport itself only ever clips a line to its width character-for-
// character — bubbles/viewport has no word-wrapping of its own, unlike the
// panels drawn with lipgloss's Width elsewhere in this package — so a raw
// paragraph handed to SetContent loses everything past the first line's worth
// of characters rather than reflowing onto more lines. Wrapping is done here,
// on the plain paragraph, with colour applied to the already-wrapped block
// afterward so nothing about the wrapping has to reason about escape codes.
func (a *About) body() string {
	width := max(a.vp.Width, 20)
	wrap := lipgloss.NewStyle().Width(width)

	// traceEventsIndex is which paragraph in aboutNotes introduces the
	// tracer — the box of event kinds goes right after it, not tied to any
	// other position in the prose.
	const traceEventsIndex = 4

	var blocks []string
	for i, p := range aboutNotes {
		blocks = append(blocks, dimStyle.Render(wrap.Render(p)))
		if i == traceEventsIndex {
			blocks = append(blocks, a.traceEventsBox(width))
		}
	}
	return strings.Join(blocks, "\n\n")
}

// traceEventsBox renders traceEventDocs as its own bordered panel — a
// reference list, set apart from the surrounding prose the same way the
// welcome screen sets "This machine" apart from its own menu.
func (a *About) traceEventsBox(width int) string {
	nameWidth := 0
	for _, e := range traceEventDocs {
		nameWidth = max(nameWidth, len(e.kind))
	}
	// Border (2) and padding (4) come out of width on top of what's given,
	// same arithmetic as every other panel in this package — see layout()'s
	// own comment on the panel wrapping this box.
	descWidth := max(width-2-4-nameWidth-2, 10)
	wrap := lipgloss.NewStyle().Width(descWidth)

	rows := []string{heading("What the tracer records"), ""}
	for _, e := range traceEventDocs {
		name := keyStyle.Render(fmt.Sprintf("%-*s", nameWidth, e.kind))
		lines := strings.Split(wrap.Render(e.desc), "\n")
		row := name + "  " + dimStyle.Render(lines[0])
		for _, l := range lines[1:] {
			row += "\n" + strings.Repeat(" ", nameWidth+2) + dimStyle.Render(l)
		}
		rows = append(rows, row)
	}
	return panelStyle.Width(width - 2).Render(strings.Join(rows, "\n"))
}

func (a *About) View() string {
	bar := a.bar()
	inner := max(a.w-2*screenMargin, minSpecsWidth)

	panel := panelStyle.Width(inner - 2).Height(a.vp.Height).Render(a.vp.View())
	body := lipgloss.JoinVertical(lipgloss.Left,
		header("GoLlama", "what is this", inner), "", panel)

	return screen(a.w, a.h, body, bar)
}

func (a *About) bar() string {
	keys := []string{
		keyStyle.Render("↑↓") + dimStyle.Render(" scroll"),
		keyStyle.Render("b") + dimStyle.Render(" back"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	return toolbar(a.w, strings.Join(keys, dimStyle.Render(" · ")), "")
}
