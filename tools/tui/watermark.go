// The fifth kind of screen: compare a prompt generated plain against the
// same prompt generated through a watermarking tournament sampler, and show
// what a detector reads off each.
//
// Unlike Ablation/Attention/Attribution/Logit Lens, this isn't a window onto
// this model's internals — it's a demo of a technique layered on top of
// ordinary sampling, so it gets its own screen rather than a fifth Inspect
// view. Structurally it's closer to Chat: one prompt in, reached the same
// way every other tool is (welcome menu, then the picker), but where Chat
// streams a single reply this waits for one finished comparison at a time —
// there's nothing worth streaming token by token when the point is the
// finished statistic, not watching it arrive.
//
// Same boundary as everywhere else in this package: nothing here imports
// engine/model. WatermarkScore is this package's own copy of what the real
// detector reports, filled in by whatever WatermarkEngine the caller
// supplies.
package tui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// WatermarkScore mirrors watermark.Score without this package depending on
// tools/watermark (which itself depends on engine/model).
type WatermarkScore struct {
	Positions int
	MeanG     float64
	Z         float64
}

// Messages the engine goroutine behind a Watermark screen sends.
type (
	// WatermarkReady says the checkpoint is loaded and a prompt can be
	// submitted. Params describes the watermark's fixed shape — key,
	// context size, tournament depth — so the screen can show it once
	// rather than repeat it with every result.
	WatermarkReady struct{ Params string }
	// WatermarkResult is one finished comparison: both generations, and the
	// detector's read on each.
	WatermarkResult struct {
		Plain, Marked           string
		PlainScore, MarkedScore WatermarkScore
	}
	WatermarkErr struct{ Err error }
)

// WatermarkOutcome is how the watermark screen ended.
type WatermarkOutcome int

const (
	WatermarkBack WatermarkOutcome = iota
	WatermarkQuit
)

type watermarkPhase int

const (
	watermarkLoading watermarkPhase = iota
	watermarkEditing
	watermarkRunning
)

// Watermark is the bubbletea model for the watermark-comparison screen.
type Watermark struct {
	events <-chan tea.Msg
	reqs   chan<- string

	phase   watermarkPhase
	params  string // set once, from WatermarkReady
	input   textinput.Model
	spin    spinner.Model
	err     error
	outcome WatermarkOutcome

	result WatermarkResult
	have   bool // whether result holds a finished comparison yet

	w, h int
}

var _ tea.Model = (*Watermark)(nil)

// NewWatermark builds the screen. reqs should be buffered by at least one —
// same reasoning as NewChat's — since submit sends on it from a tea.Cmd
// rather than blocking Update.
func NewWatermark(events <-chan tea.Msg, reqs chan<- string, prompt string) *Watermark {
	in := textinput.New()
	in.Placeholder = "a prompt to compare"
	in.SetValue(prompt)
	in.CharLimit = 300
	in.Prompt = "❯ "
	in.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Dot

	return &Watermark{events: events, reqs: reqs, phase: watermarkLoading, input: in, spin: sp, w: 100, h: 32}
}

// Outcome reports how the screen ended. Valid once it has finished.
func (w *Watermark) Outcome() WatermarkOutcome { return w.outcome }

func (w *Watermark) Init() tea.Cmd { return tea.Batch(waitForWatermark(w.events), w.spin.Tick) }

// waitForWatermark is waitForChat's counterpart for this screen's channel.
func waitForWatermark(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return WatermarkErr{Err: watermarkClosedErr{}}
		}
		return msg
	}
}

type watermarkClosedErr struct{}

func (watermarkClosedErr) Error() string { return "the engine stopped responding" }

func (w *Watermark) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		w.w, w.h = msg.Width, msg.Height
		return w, nil
	case spinner.TickMsg:
		if w.phase != watermarkLoading && w.phase != watermarkRunning {
			return w, nil
		}
		var cmd tea.Cmd
		w.spin, cmd = w.spin.Update(msg)
		return w, cmd
	case WatermarkReady:
		w.params = msg.Params
		w.phase = watermarkEditing
		return w, waitForWatermark(w.events)
	case WatermarkResult:
		w.result, w.have = msg, true
		w.phase = watermarkEditing
		return w, waitForWatermark(w.events)
	case WatermarkErr:
		w.err = msg.Err
		w.phase = watermarkEditing
		return w, waitForWatermark(w.events)
	case tea.KeyMsg:
		return w.handleKey(msg)
	}
	return w, nil
}

func (w *Watermark) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		w.outcome = WatermarkQuit
		return w, done
	case "esc":
		w.outcome = WatermarkBack
		return w, done
	case "enter":
		return w, w.submit()
	}
	if w.phase != watermarkEditing {
		return w, nil
	}
	var cmd tea.Cmd
	w.input, cmd = w.input.Update(msg)
	return w, cmd
}

// submit sends the input's contents to the engine and starts a run. Mirrors
// Chat.submit: the send happens inside the returned tea.Cmd rather than here,
// so a slow engine can't stall the render loop.
func (w *Watermark) submit() tea.Cmd {
	text := strings.TrimSpace(w.input.Value())
	if text == "" || w.phase != watermarkEditing {
		return nil
	}
	w.phase, w.err = watermarkRunning, nil
	reqs := w.reqs
	return tea.Batch(func() tea.Msg { reqs <- text; return nil }, w.spin.Tick)
}

// --- view ---------------------------------------------------------------

func (w *Watermark) View() string {
	bar := w.bar()
	inner := max(w.w-2*screenMargin, minSpecsWidth)
	panel := stretch(panelStyle.Width(inner-2), bodyRows(w.h, bar)-lipgloss.Height(w.header()), w.content(inner-4))
	body := lipgloss.JoinVertical(lipgloss.Left, w.header(), panel)
	return screen(w.w, w.h, body, bar)
}

func (w *Watermark) header() string {
	lines := []string{titleStyle.Render(" GoLlama watermark ") + dimStyle.Render(" "+w.params)}
	switch w.phase {
	case watermarkLoading:
		lines = append(lines, " "+w.spin.View()+" "+dimStyle.Render("loading…"))
	case watermarkRunning:
		lines = append(lines, " "+w.spin.View()+" "+dimStyle.Render(fmt.Sprintf("comparing %q…", w.input.Value())))
	default:
		lines = append(lines, " "+w.input.View())
	}
	return strings.Join(lines, "\n") + "\n"
}

func (w *Watermark) content(width int) string {
	if w.err != nil {
		return " " + warnStyle.Render("error: "+w.err.Error())
	}
	if !w.have {
		return " " + dimStyle.Render("type a prompt and press enter to compare plain sampling against the watermark.")
	}
	wrap := lipgloss.NewStyle().Width(width)
	return strings.Join([]string{
		w.side("plain", w.result.Plain, w.result.PlainScore, wrap),
		"",
		w.side("watermarked", w.result.Marked, w.result.MarkedScore, wrap),
	}, "\n")
}

// side renders one half of the comparison: its label and detector reading on
// one line, its generated text wrapped underneath.
func (w *Watermark) side(label, text string, score WatermarkScore, wrap lipgloss.Style) string {
	head := fmt.Sprintf(" %s  mean g %.3f  z %s",
		headingStyle.Render(label), score.MeanG, w.zLabel(score.Z))
	return head + "\n" + dimStyle.Render(wrap.Render(strings.TrimSpace(text)))
}

// zLabel colors the z-score itself: past the detection line it's the same
// warning color a real detector's "flagged" verdict would be, since that's
// the one number this whole screen is building up to.
func (w *Watermark) zLabel(z float64) string {
	label := fmt.Sprintf("%.2f", z)
	if z >= 4 {
		return warnStyle.Render(label)
	}
	return valueStyle.Render(label)
}

func (w *Watermark) bar() string {
	keys := []string{"enter compare"}
	if w.err != nil {
		keys = []string{"enter retry"}
	}
	keys = append(keys, "esc back", "ctrl+c quit")
	return toolbar(w.w, strings.Join(keys, dimStyle.Render(" · ")), "")
}
