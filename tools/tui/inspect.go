// The fourth kind of screen: watch a forward pass. This used to be a wholly
// separate program, cmd/inspect — its own binary, its own checkpoint flag,
// discoverable only by reading a footer or typing `go run ./cmd/inspect`. It's
// a screen here instead, reached the same way Chat is: pick a tool from the
// welcome menu, pick a model on the picker, land here already loaded.
//
// One screen serves five views (ablation, attention, attribution, logit lens,
// stages) rather than one screenID per tool, because they already share one
// engine goroutine, one KV cache, and one step history — switching views is
// free, it doesn't re-run inference, and ablation specifically depends on
// comparing against the same baseline run the other views are browsing.
// Picking a different tool from the welcome menu opens this screen with a
// different initial view; the in-screen 1-5/tab switcher is still there to
// cross-check another view of the same run without leaving it.
//
// This file is the one exception to "nothing in tools/tui imports engine/
// model or engine/tokenizer": it imports tools/trace for Trace/Event, the
// passive data format both the live engine and file replay already share.
// That's a deliberate narrowing of the invariant, not a full severing —
// trace has no dependency on a live model the way model.GPT or
// tokenizer.Tokenizer would, so reading one doesn't require this package to
// know how to run a forward pass. The live tokenization-preview feature is
// the one case that would otherwise need a real tokenizer here; it's routed
// through InspectRequest{Preview: true}/InspectPreview instead, so the
// tokenizer itself stays on the engine side of the boundary, same as
// everything else this screen shows.
package tui

import (
	"context"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

// HeadRef identifies one query attention head to ablate — a tui-local mirror
// of model.HeadRef, the same way catalog.go's Arch mirrors model.GPTConfig.
// Two ints, trivially copyable, no dependency on engine/model.
type HeadRef struct{ Layer, Head int }

// InspectEngine is how the root screen turns a chosen model into a running
// inspection session: read requests off reqs, write InspectReady/InspectStep/
// InspectPreview/InspectRunDone/InspectErr onto events, and stop when ctx is
// cancelled. Mirrors Engine's shape exactly, for the same reason — the screen
// owns the frame, whoever calls Start owns what "run a pass" means.
type InspectEngine func(ctx context.Context, dir string, reqs <-chan InspectRequest, events chan<- tea.Msg)

// InspectRequest is one thing to ask the engine for: either run a prompt
// (optionally with some heads ablated), or just tokenize it for the live
// preview while the prompt is still being typed.
type InspectRequest struct {
	Prompt    string
	MaxTokens int
	// Ablate, when non-empty, forces the listed attention heads' output to
	// zero for the whole run.
	Ablate []HeadRef
	// Preview means "tokenize Prompt and reply with InspectPreview" — don't
	// run the model at all. Encoding a short prompt is microseconds, so this
	// can be fired on every keystroke without visibly lagging typing.
	Preview bool
}

// Messages the engine goroutine sends to the UI.
type (
	// InspectStatus is progress text shown while work is in flight.
	InspectStatus string
	// InspectReady says the checkpoint is loaded and prompts can be
	// submitted.
	InspectReady struct{ Info trace.ModelInfo }
	// InspectPreview answers a Preview request: the prompt's tokens, decoded
	// and with whitespace already made visible.
	InspectPreview struct{ Tokens []string }
	// InspectStep is one completed traced pass: the prefill, or one
	// generated token.
	InspectStep struct {
		Label   string
		Tr      *trace.Trace
		Ablated bool // true when this step came from a request with Ablate set
	}
	// InspectRunDone says a whole prompt finished, so the UI can go back to
	// browsing.
	InspectRunDone struct{}
	// InspectErr carries a failure from the engine side, including the
	// engine goroutine ending unexpectedly (see waitForInspect).
	InspectErr struct{ Err error }
)

// errInspectClosed is what a closed events channel is reported as, the same
// convention chat.go's errClosed uses: the engine only closes it after a
// failure it already sent as an InspectErr, so this only surfaces if that
// message was somehow missed.
var errInspectClosed = inspectClosedErr{}

type inspectClosedErr struct{}

func (inspectClosedErr) Error() string { return "the engine stopped responding" }

// waitForInspect turns the next message off ch into a bubbletea command; it
// has to be reissued after every message to keep listening.
func waitForInspect(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return InspectErr{Err: errInspectClosed}
		}
		return msg
	}
}

// InspectOutcome is how the inspect screen ended.
type InspectOutcome int

const (
	// InspectBack means they backed out to pick a different model — esc.
	InspectBack InspectOutcome = iota
	// InspectQuit means they left the program entirely.
	InspectQuit
)

// inspectView is which of the five panes is showing.
type inspectView int

const (
	viewAblation    inspectView = iota // baseline vs. one head forced to zero
	viewAttention                      // what each head looked at
	viewAttribution                    // which components moved the answer
	viewLens                           // how the prediction forms, layer by layer
	viewStages
	numInspectViews
)

func (v inspectView) String() string {
	switch v {
	case viewAblation:
		return "ablation"
	case viewAttention:
		return "attention"
	case viewAttribution:
		return "attribution"
	case viewLens:
		return "logit lens"
	default:
		return "stages"
	}
}

// toolInitialView maps a welcome-menu Tool to the view Inspect should open
// on. ToolChat has no mapping — it opens Chat, never Inspect.
func toolInitialView(t Tool) inspectView {
	switch t {
	case ToolAttention:
		return viewAttention
	case ToolAttribution:
		return viewAttribution
	case ToolLens:
		return viewLens
	default:
		return viewAblation
	}
}

// inspectPhase is what the screen is doing, which decides where a keystroke
// goes.
type inspectPhase int

const (
	inspectLoading  inspectPhase = iota // waiting for the checkpoint
	inspectEditing                      // typing a prompt
	inspectRunning                      // inference in flight
	inspectBrowsing                     // reading results
)

// inspectStep is one traced forward pass: the prefill, or one generated
// token.
type inspectStep struct {
	label   string
	tr      *trace.Trace
	layers  map[int][]trace.Event
	outside []trace.Event
	lens    []trace.Event
}

func newInspectStep(label string, tr *trace.Trace) inspectStep {
	layers, outside := tr.ByLayer()
	return inspectStep{
		label: label, tr: tr,
		layers: layers, outside: outside,
		lens: tr.Kind(trace.KindLogitLens),
	}
}

// Inspect is the bubbletea model for the pass-inspection screen.
type Inspect struct {
	steps []inspectStep
	cur   int // which step is displayed

	// Live mode. Both nil when replaying a file.
	events <-chan tea.Msg
	reqs   chan<- InspectRequest

	phase     inspectPhase
	input     textinput.Model
	info      trace.ModelInfo
	maxTokens int
	status    string
	err       error
	outcome   InspectOutcome

	// preview is the last token-preview response, rendered alongside the
	// prompt field while editing. It can lag the input by one keystroke —
	// the round trip to the engine is asynchronous — which is a smaller cost
	// than giving this screen its own tokenizer.
	preview []string

	view  inspectView
	layer int
	head  int
	w, h  int

	// Ablation. ablateOn is whether the currently selected (layer, head) is
	// being forced to zero for comparison; ablateSteps holds that shadow
	// run's steps, index-aligned with steps (ablateSteps[i] is the ablated
	// version of steps[i]). lastPrompt is what submit() last sent, so the
	// ablated re-run can resubmit the same text without touching the input.
	ablateOn    bool
	ablateSteps []inspectStep
	lastPrompt  string
}

var _ tea.Model = (*Inspect)(nil)

// NewInspectFile builds a file-mode screen: one step already loaded from a
// replayed trace, nothing to run.
func NewInspectFile(tr *trace.Trace) *Inspect {
	a := &Inspect{w: 100, h: 30, phase: inspectBrowsing, maxTokens: 3, view: viewLens}
	a.info = tr.Header.Config
	a.addStep(newInspectStep("trace", tr))
	return a
}

// NewInspectLive builds an interactive screen: type a prompt, run it, browse
// the trace. initial is which view to open on — set from the tool picked on
// the welcome menu, so choosing "Attention" opens straight to the attention
// grid rather than always landing on the logit lens.
func NewInspectLive(events <-chan tea.Msg, reqs chan<- InspectRequest, prompt string, maxTokens int, initial inspectView) *Inspect {
	in := textinput.New()
	in.Placeholder = "type anything and press enter"
	in.SetValue(prompt) // usually empty; -prompt pre-fills it for scripted runs
	in.CharLimit = 512
	in.Width = 60
	in.Prompt = "prompt ❯ "

	return &Inspect{
		w: 100, h: 30,
		events: events, reqs: reqs,
		phase: inspectLoading, input: in, maxTokens: maxTokens,
		status: "starting…", view: initial,
	}
}

// Outcome reports how the screen ended. Valid once it has finished.
func (a *Inspect) Outcome() InspectOutcome { return a.outcome }

func (a *Inspect) live() bool { return a.events != nil }

func (a *Inspect) addStep(s inspectStep) {
	a.steps = append(a.steps, s)
	// Follow the newest step, and start on the last layer where the prediction
	// has settled — arrowing up from there walks the reasoning backwards.
	a.cur = len(a.steps) - 1
	a.layer = max(0, s.tr.Header.Config.NLayer-1)
}

// active is the step being displayed, or nil before the first one lands.
func (a *Inspect) active() *inspectStep {
	if a.cur < 0 || a.cur >= len(a.steps) {
		return nil
	}
	return &a.steps[a.cur]
}

func (a *Inspect) config() trace.ModelInfo {
	if s := a.active(); s != nil {
		return s.tr.Header.Config
	}
	return a.info
}

func (a *Inspect) Init() tea.Cmd {
	if a.live() {
		return tea.Batch(waitForInspect(a.events), textinput.Blink)
	}
	return nil
}

func (a *Inspect) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.w, a.h = msg.Width, msg.Height
		a.input.Width = max(20, a.w-20)

	case InspectStatus:
		a.status = string(msg)
		return a, waitForInspect(a.events)

	case InspectReady:
		a.info = msg.Info
		a.phase = inspectEditing
		a.status = ""
		a.input.Focus()
		return a, tea.Batch(waitForInspect(a.events), textinput.Blink, a.requestPreview())

	case InspectPreview:
		a.preview = msg.Tokens
		return a, waitForInspect(a.events)

	case InspectStep:
		switch {
		case msg.Ablated && a.ablateOn:
			// A shadow run for comparison, not the baseline being browsed —
			// append alongside it rather than through addStep, which would
			// reset the cursor/layer the baseline is currently showing.
			a.ablateSteps = append(a.ablateSteps, newInspectStep(msg.Label, msg.Tr))
		case msg.Ablated:
			// Ablation was toggled off again before this step landed; drop it.
		default:
			a.addStep(newInspectStep(msg.Label, msg.Tr))
		}
		a.status = ""
		return a, waitForInspect(a.events)

	case InspectRunDone:
		a.phase = inspectBrowsing
		a.status = ""
		return a, waitForInspect(a.events)

	case InspectErr:
		a.err = msg.Err
		a.phase = inspectBrowsing
		return a, nil

	case tea.KeyMsg:
		return a.handleKey(msg)
	}

	if a.phase == inspectEditing {
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		return a, cmd
	}
	return a, nil
}

func (a *Inspect) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	// Quitting always works, including mid-run.
	if msg.Type == tea.KeyCtrlC {
		a.outcome = InspectQuit
		return a, done
	}

	if a.phase == inspectEditing {
		switch msg.Type {
		case tea.KeyEnter:
			return a, a.submit()
		case tea.KeyEsc:
			// Only leave the field if there's something to look at.
			if len(a.steps) > 0 {
				a.phase = inspectBrowsing
				a.input.Blur()
			}
			return a, nil
		}
		before := a.input.Value()
		var cmd tea.Cmd
		a.input, cmd = a.input.Update(msg)
		if a.input.Value() != before {
			if a.input.Value() == "" {
				a.preview = nil
			} else {
				cmd = tea.Batch(cmd, a.requestPreview())
			}
		}
		return a, cmd
	}

	cfg := a.config()
	switch msg.String() {
	case "esc":
		a.outcome = InspectBack
		return a, done
	case "q":
		a.outcome = InspectQuit
		return a, done
	case "i", "/":
		if a.live() {
			a.phase = inspectEditing
			a.input.Focus()
			return a, tea.Batch(textinput.Blink, a.requestPreview())
		}
	case "tab", "]":
		a.view = (a.view + 1) % numInspectViews
	case "shift+tab", "[":
		a.view = (a.view + numInspectViews - 1) % numInspectViews
	case "1":
		a.view = viewAblation
	case "2":
		a.view = viewAttention
	case "3":
		a.view = viewAttribution
	case "4":
		a.view = viewLens
	case "5":
		a.view = viewStages
	case "a":
		return a, a.toggleAblate()
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
func (a *Inspect) submit() tea.Cmd {
	prompt := strings.TrimRight(a.input.Value(), "\n")
	if prompt == "" || a.reqs == nil {
		return nil
	}
	a.steps = nil
	a.cur = 0
	a.err = nil
	a.phase = inspectRunning
	a.status = "encoding…"
	a.input.Blur()

	// A fresh prompt invalidates whatever ablation comparison was showing.
	a.ablateOn = false
	a.ablateSteps = nil
	a.lastPrompt = prompt

	req := InspectRequest{Prompt: prompt, MaxTokens: a.maxTokens}
	return func() tea.Msg {
		a.reqs <- req
		return nil
	}
}

// requestPreview asks the engine to tokenize whatever's currently in the
// prompt field, for the live "N tokens: ..." line — see InspectPreview.
func (a *Inspect) requestPreview() tea.Cmd {
	if !a.live() || a.reqs == nil || a.input.Value() == "" {
		return nil
	}
	req := InspectRequest{Prompt: a.input.Value(), Preview: true}
	return func() tea.Msg {
		a.reqs <- req
		return nil
	}
}

// toggleAblate turns ablation of the currently selected (layer, head) on or
// off. Turning it on fires a fresh request with that head forced to zero, so
// viewAblation has a shadow run to compare against the baseline already on
// screen; turning it off just drops that shadow run.
func (a *Inspect) toggleAblate() tea.Cmd {
	if a.ablateOn {
		a.ablateOn = false
		a.ablateSteps = nil
		return nil
	}
	if !a.live() || a.lastPrompt == "" || a.reqs == nil {
		return nil
	}
	a.ablateOn = true
	a.ablateSteps = nil

	req := InspectRequest{
		Prompt:    a.lastPrompt,
		MaxTokens: a.maxTokens,
		Ablate:    []HeadRef{{Layer: a.layer, Head: a.head}},
	}
	return func() tea.Msg {
		a.reqs <- req
		return nil
	}
}

func (a *Inspect) View() string {
	if a.err != nil {
		return a.header() + "\n " + warnStyle.Render("error: "+a.err.Error()) +
			"\n\n " + dimStyle.Render("i to edit the prompt · esc to go back") + "\n"
	}
	if a.active() == nil {
		return a.header() + "\n " + dimStyle.Render(a.status) + "\n" + a.footer()
	}

	body := ""
	switch a.view {
	case viewAblation:
		body = a.ablationView()
	case viewAttention:
		body = a.attentionView()
	case viewAttribution:
		body = a.attributionView()
	case viewLens:
		body = a.lensView()
	case viewStages:
		body = a.stagesView()
	}
	return lipgloss.JoinVertical(lipgloss.Left, a.header(), body, a.footer())
}

// --- chrome -----------------------------------------------------------------

func (a *Inspect) header() string {
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

	tabs := make([]string, numInspectViews)
	for v := inspectView(0); v < numInspectViews; v++ {
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

// promptLine is the input field, plus the live tokenization preview so you
// can see how the prompt will actually be split before running it.
func (a *Inspect) promptLine() string {
	switch a.phase {
	case inspectLoading:
		return " " + dimStyle.Render(a.status)
	case inspectEditing:
		line := " " + a.input.View()
		if len(a.preview) > 0 {
			line += "\n " + dimStyle.Render(fmt.Sprintf("%d tokens: ", len(a.preview))) +
				keyStyle.Render(formatTokenPreview(a.preview, 12))
		}
		return line + dimStyle.Render(fmt.Sprintf("\n enter to run · %d tokens to generate (+/-)",
			a.maxTokens))
	case inspectRunning:
		return " " + dimStyle.Render(fmt.Sprintf("%q → %s", a.input.Value(), a.status))
	default:
		return a.resultLine(a.input.Value())
	}
}

func (a *Inspect) resultLine(prompt string) string {
	line := fmt.Sprintf(" %s", keyStyle.Render(fmt.Sprintf("%q", prompt)))
	if out := a.generated(); out != "" {
		line += dimStyle.Render("  →  ") + headingStyle.Render(fmt.Sprintf("%q", out))
	}
	return line
}

// generated joins the labels of every decode step, which is the text the
// model produced.
func (a *Inspect) generated() string {
	var b strings.Builder
	for _, s := range a.steps {
		if s.label != "prefill" && s.label != "trace" {
			b.WriteString(s.label)
		}
	}
	return b.String()
}

func (a *Inspect) stepStrip() string {
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

func (a *Inspect) footer() string {
	var keys []string
	if a.phase == inspectEditing {
		keys = []string{"enter run", "+/- tokens", "esc browse", "ctrl+c quit"}
	} else {
		keys = []string{"↑↓ layer", "←→ head", "tab view"}
		if len(a.steps) > 1 {
			keys = append(keys, "n/p step")
		}
		if a.live() {
			keys = append(keys, "a ablate head", "i prompt")
		}
		keys = append(keys, "esc back", "q quit")
	}
	return "\n" + dimStyle.Render(" "+strings.Join(keys, " · "))
}

// finalPrediction is the lens readout past the last block — the model's actual
// output, recorded so intermediate layers have something to be compared against.
func (a *Inspect) finalPrediction() *trace.Candidate {
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
func (a *Inspect) bodyHeight() int {
	chrome := 7
	if len(a.steps) > 1 {
		chrome++
	}
	if a.phase == inspectEditing {
		chrome += 2 // tokenization preview and the hint line
	}
	return max(3, a.h-chrome)
}

// formatTokenPreview renders the first n token strings with a separator, the
// way the prompt's tokenization is shown as you type. tokens already have
// their whitespace made visible — see sanitize — since that happens on the
// engine side, which is the one place with a real tokenizer to decode with.
func formatTokenPreview(tokens []string, n int) string {
	if n > len(tokens) {
		n = len(tokens)
	}
	s := strings.Join(tokens[:n], "|")
	if len(tokens) > n {
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
