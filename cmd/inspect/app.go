package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

type view int

const (
	viewLens        view = iota // how the prediction forms, layer by layer
	viewAttention               // what each head looked at
	viewAttribution             // which components moved the answer
	viewStages
	numViews
)

func (v view) String() string {
	switch v {
	case viewLens:
		return "logit lens"
	case viewAttention:
		return "attention"
	case viewAttribution:
		return "attribution"
	default:
		return "stages"
	}
}

// phase is what the app is doing, which decides where keystrokes go.
type phase int

const (
	phaseLoading  phase = iota // waiting for the checkpoint
	phaseEditing               // typing a prompt
	phaseRunning               // inference in flight
	phaseBrowsing              // reading results
)

// step is one traced forward pass: the prefill, or one generated token.
type step struct {
	label   string
	tr      *trace.Trace
	layers  map[int][]trace.Event
	outside []trace.Event
	lens    []trace.Event
}

func newStep(label string, tr *trace.Trace) step {
	layers, outside := tr.ByLayer()
	return step{
		label: label, tr: tr,
		layers: layers, outside: outside,
		lens: tr.Kind(trace.KindLogitLens),
	}
}

type app struct {
	steps []step
	cur   int // which step is displayed

	// Live mode. Both nil when replaying a file.
	events <-chan tea.Msg
	reqs   chan<- request

	phase     phase
	input     textinput.Model
	tok       *tokenizer.Tokenizer // for showing tokenization as you type
	info      trace.ModelInfo
	maxTokens int
	status    string
	err       error

	view  view
	layer int
	head  int
	w, h  int
}

// newApp builds a file-mode app: one step, nothing to run.
func newApp(tr *trace.Trace) *app {
	a := &app{w: 100, h: 30, phase: phaseBrowsing, maxTokens: 3}
	a.info = tr.Header.Config
	a.addStep(newStep("trace", tr))
	return a
}

// newLiveApp builds an interactive app: type a prompt, run it, browse the trace.
func newLiveApp(events <-chan tea.Msg, reqs chan<- request, prompt string, maxTokens int) *app {
	in := textinput.New()
	in.Placeholder = "type anything and press enter"
	in.SetValue(prompt) // usually empty; -prompt pre-fills it for scripted runs
	in.CharLimit = 512
	in.Width = 60
	in.Prompt = "prompt ❯ "

	return &app{
		w: 100, h: 30,
		events: events, reqs: reqs,
		phase: phaseLoading, input: in, maxTokens: maxTokens,
		status: "starting…",
	}
}

func (a *app) live() bool { return a.events != nil }

func (a *app) addStep(s step) {
	a.steps = append(a.steps, s)
	// Follow the newest step, and start on the last layer where the prediction
	// has settled — arrowing up from there walks the reasoning backwards.
	a.cur = len(a.steps) - 1
	a.layer = max(0, s.tr.Header.Config.NLayer-1)
}

// active is the step being displayed, or nil before the first one lands.
func (a *app) active() *step {
	if a.cur < 0 || a.cur >= len(a.steps) {
		return nil
	}
	return &a.steps[a.cur]
}

func (a *app) config() trace.ModelInfo {
	if s := a.active(); s != nil {
		return s.tr.Header.Config
	}
	return a.info
}

func (a *app) Init() tea.Cmd {
	if a.live() {
		return tea.Batch(waitFor(a.events), textinput.Blink)
	}
	return nil
}

func (a *app) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.w, a.h = msg.Width, msg.Height
		a.input.Width = max(20, a.w-20)

	case statusMsg:
		a.status = string(msg)
		return a, waitFor(a.events)

	case readyMsg:
		a.tok, a.info = msg.tok, msg.cfg
		a.phase = phaseEditing
		a.status = ""
		a.input.Focus()
		return a, tea.Batch(waitFor(a.events), textinput.Blink)

	case stepMsg:
		a.addStep(newStep(msg.label, msg.tr))
		a.status = ""
		return a, waitFor(a.events)

	case runDoneMsg:
		a.phase = phaseBrowsing
		a.status = ""
		return a, waitFor(a.events)

	case errMsg:
		a.err = msg.err
		a.phase = phaseBrowsing
		return a, nil

	case doneMsg:
		if a.phase == phaseRunning || a.phase == phaseLoading {
			a.phase = phaseBrowsing
		}
		a.status = ""
		return a, nil

	case tea.KeyMsg:
		return a.handleKey(msg)
	}

	if a.phase == phaseEditing {
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		return a, cmd
	}
	return a, nil
}

func (a *app) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Quitting always works, including mid-run.
	switch msg.Type {
	case tea.KeyCtrlC:
		return a, tea.Quit
	}

	if a.phase == phaseEditing {
		switch msg.Type {
		case tea.KeyEnter:
			return a, a.submit()
		case tea.KeyEsc:
			// Only leave the field if there's something to look at.
			if len(a.steps) > 0 {
				a.phase = phaseBrowsing
				a.input.Blur()
			}
			return a, nil
		}
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		return a, cmd
	}

	cfg := a.config()
	switch msg.String() {
	case "q", "esc":
		return a, tea.Quit
	case "i", "/":
		if a.live() {
			a.phase = phaseEditing
			a.input.Focus()
			return a, textinput.Blink
		}
	case "tab", "]":
		a.view = (a.view + 1) % numViews
	case "shift+tab", "[":
		a.view = (a.view + numViews - 1) % numViews
	case "1":
		a.view = viewLens
	case "2":
		a.view = viewAttention
	case "3":
		a.view = viewAttribution
	case "4":
		a.view = viewStages
	case "up", "k":
		a.layer = max(0, a.layer-1)
	case "down", "j":
		a.layer = min(max(0, cfg.NLayer-1), a.layer+1)
	case "g", "home":
		a.layer = 0
	case "G", "end":
		a.layer = max(0, cfg.NLayer-1)
	case "left", "h":
		a.head = max(0, a.head-1)
	case "right", "l":
		a.head = min(max(0, cfg.NHead-1), a.head+1)
	case "n", ".", "pgdown":
		a.cur = min(len(a.steps)-1, a.cur+1)
	case "p", ",", "pgup":
		a.cur = max(0, a.cur-1)
	case "+", "=":
		a.maxTokens = min(32, a.maxTokens+1)
	case "-", "_":
		a.maxTokens = max(0, a.maxTokens-1)
	}
	return a, nil
}

// submit sends the typed prompt to the engine and clears the previous run.
func (a *app) submit() tea.Cmd {
	prompt := strings.TrimRight(a.input.Value(), "\n")
	if prompt == "" || a.reqs == nil {
		return nil
	}
	a.steps = nil
	a.cur = 0
	a.err = nil
	a.phase = phaseRunning
	a.status = "encoding…"
	a.input.Blur()

	req := request{prompt: prompt, maxTokens: a.maxTokens}
	return func() tea.Msg {
		a.reqs <- req
		return nil
	}
}

func (a *app) View() string {
	if a.err != nil {
		return a.header() + "\n " + errStyle.Render("error: "+a.err.Error()) +
			"\n\n " + dimStyle.Render("i to edit the prompt · q to quit") + "\n"
	}
	if a.active() == nil {
		return a.header() + "\n " + dimStyle.Render(a.status) + "\n" + a.footer()
	}

	body := ""
	switch a.view {
	case viewLens:
		body = a.lensView()
	case viewAttention:
		body = a.attentionView()
	case viewAttribution:
		body = a.attributionView()
	case viewStages:
		body = a.stagesView()
	}
	return lipgloss.JoinVertical(lipgloss.Left, a.header(), body, a.footer())
}

// --- chrome -----------------------------------------------------------------

func (a *app) header() string {
	c := a.config()
	title := titleStyle.Render(" GoLlama inspect ")
	meta := ""
	if c.NLayer > 0 {
		meta = dimStyle.Render(fmt.Sprintf(" %d layers · %d heads (%d kv) · %d dims",
			c.NLayer, c.NHead, c.NKVHead, c.NEmbed))
	}
	lines := []string{title + meta}

	if a.live() {
		lines = append(lines, a.promptLine())
	} else if s := a.active(); s != nil {
		lines = append(lines, a.resultLine(s.tr.Header.Prompt))
	}

	if len(a.steps) > 1 {
		lines = append(lines, a.stepStrip())
	}

	tabs := make([]string, numViews)
	for v := view(0); v < numViews; v++ {
		label := fmt.Sprintf(" %d %s ", v+1, v)
		if v == a.view {
			tabs[v] = activeTabStyle.Render(label)
		} else {
			tabs[v] = tabStyle.Render(label)
		}
	}
	lines = append(lines, " "+strings.Join(tabs, " "), "")
	return strings.Join(lines, "\n")
}

// promptLine is the input field, plus live tokenization so you can see how the
// prompt will actually be split before running it.
func (a *app) promptLine() string {
	switch a.phase {
	case phaseLoading:
		return " " + dimStyle.Render(a.status)
	case phaseEditing:
		line := " " + a.input.View()
		if a.tok != nil {
			if ids := a.tok.Encode(a.input.Value()); len(ids) > 0 {
				line += "\n " + dimStyle.Render(fmt.Sprintf("%d tokens: ", len(ids))) +
					keyStyle.Render(tokenPreview(a.tok, ids, 12))
			}
		}
		return line + dimStyle.Render(fmt.Sprintf("\n enter to run · %d tokens to generate (+/-)",
			a.maxTokens))
	case phaseRunning:
		return " " + dimStyle.Render(fmt.Sprintf("%q → %s", a.input.Value(), a.status))
	default:
		return a.resultLine(a.input.Value())
	}
}

func (a *app) resultLine(prompt string) string {
	line := fmt.Sprintf(" %s", keyStyle.Render(fmt.Sprintf("%q", prompt)))
	if out := a.generated(); out != "" {
		line += dimStyle.Render("  →  ") + hotStyle.Render(fmt.Sprintf("%q", out))
	}
	return line
}

// generated joins the labels of every decode step, which is the text the model
// produced.
func (a *app) generated() string {
	var b strings.Builder
	for _, s := range a.steps {
		if s.label != "prefill" && s.label != "trace" {
			b.WriteString(s.label)
		}
	}
	return b.String()
}

func (a *app) stepStrip() string {
	var b strings.Builder
	b.WriteString(" steps ")
	for i, st := range a.steps {
		label := fmt.Sprintf(" %s ", sanitize(st.label))
		if i == a.cur {
			b.WriteString(activeTabStyle.Render(label))
		} else {
			b.WriteString(tabStyle.Render(label))
		}
	}
	if a.status != "" {
		b.WriteString(dimStyle.Render("  " + a.status))
	}
	return b.String()
}

func (a *app) footer() string {
	var keys []string
	if a.phase == phaseEditing {
		keys = []string{"enter run", "+/- tokens", "esc browse", "ctrl+c quit"}
	} else {
		keys = []string{"↑↓ layer", "←→ head", "tab view"}
		if len(a.steps) > 1 {
			keys = append(keys, "n/p step")
		}
		if a.live() {
			keys = append(keys, "i prompt")
		}
		keys = append(keys, "q quit")
	}
	return "\n" + dimStyle.Render(" "+strings.Join(keys, " · "))
}

// finalPrediction is the lens readout past the last block — the model's actual
// output, recorded so intermediate layers have something to be compared against.
func (a *app) finalPrediction() *trace.Candidate {
	s := a.active()
	if s == nil {
		return nil
	}
	for _, e := range s.lens {
		if e.Layer >= s.tr.Header.Config.NLayer && len(e.Top) > 0 {
			return &e.Top[0]
		}
	}
	return nil
}

// bodyHeight is what's left for a view after the header and footer.
func (a *app) bodyHeight() int {
	chrome := 7
	if len(a.steps) > 1 {
		chrome++
	}
	if a.phase == phaseEditing {
		chrome += 2 // tokenization preview and the hint line
	}
	return max(3, a.h-chrome)
}

// tokenPreview renders the first n tokens with visible whitespace.
func tokenPreview(tok *tokenizer.Tokenizer, ids []int, n int) string {
	if n > len(ids) {
		n = len(ids)
	}
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = sanitize(tok.Decode([]int{ids[i]}))
	}
	s := strings.Join(parts, "|")
	if len(ids) > n {
		s += "|…"
	}
	return s
}

// window returns the slice bounds to display so that `sel` stays visible when
// there are more rows than screen.
func window(total, sel, height int) (lo, hi int) {
	if total <= height {
		return 0, total
	}
	lo = max(0, min(sel-height/2, total-height))
	return lo, lo + height
}
