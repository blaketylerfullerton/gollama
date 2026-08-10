package tui

import (
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
	"go run ./cmd/inspect opens an interactive trace viewer: type a prompt, run it, then step between generated tokens to see what each one attended to and where its answer settled. The chat screen after picking a model here does a live version of the same thing — press tab there to inspect what a streamed token leaned on and what it ranked as likely to come next.",
	"Grouped-query attention, rotary position embeddings, QK-norm, SwiGLU, RMSNorm, tied embeddings — Qwen3's architecture, matching Llama everywhere it can and diverging where it has to. None of it is optimized; all of it is meant to be legible.",
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
			return a, tea.Quit
		case "q", "ctrl+c":
			a.outcome = AboutQuit
			return a, tea.Quit
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
	paras := make([]string, len(aboutNotes))
	for i, p := range aboutNotes {
		paras[i] = dimStyle.Render(wrap.Render(p))
	}
	return strings.Join(paras, "\n\n")
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
