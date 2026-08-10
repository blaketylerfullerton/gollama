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

	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// Choice is what the user did with the welcome screen.
type Choice int

const (
	// Quit means they backed out — ctrl+c, q, or esc. The caller should exit
	// without loading anything.
	Quit Choice = iota
	// Run means they pressed enter.
	Run
)

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
	sys    sysinfo.Info
	ckpt   Checkpoint
	choice Choice
	w, h   int
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
		sys:  sys,
		ckpt: ScanCheckpoint(checkpointDir),
		w:    100, h: 32,
	}
}

// Choice reports what the user picked. Valid once the program has returned.
func (w *Welcome) Choice() Choice { return w.choice }

func (w *Welcome) Init() tea.Cmd { return nil }

func (w *Welcome) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		w.w, w.h = msg.Width, msg.Height
	case tea.KeyMsg:
		switch msg.String() {
		case "enter", " ":
			w.choice = Run
			return w, tea.Quit
		case "q", "esc", "ctrl+c":
			w.choice = Quit
			return w, tea.Quit
		}
	}
	return w, nil
}

func (w *Welcome) View() string {
	art := llama()
	specs := panelStyle.Render(w.specs())

	// Side by side when there's room for both, stacked when there isn't. The
	// threshold is the art plus the panel plus the gap between them.
	body := lipgloss.JoinHorizontal(lipgloss.Top, art, "  ", specs)
	if w.w > 0 && lipgloss.Width(body) > w.w {
		body = specs
		if w.h > len(llamaArt)+16 {
			body = lipgloss.JoinVertical(lipgloss.Left, art, "", specs)
		}
	}

	return strings.Join([]string{
		"",
		" " + titleStyle.Render("GoLlama") + " " +
			subtitleStyle.Render("a transformer you can read, one token at a time"),
		"",
		indent(body, 1),
		"",
		" " + w.footer(),
		"",
	}, "\n")
}

// specs is the right-hand column: what this machine is, then what's on disk to
// run on it.
func (w *Welcome) specs() string {
	s := w.sys
	rows := []string{
		heading("this machine"),
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
		"",
		heading("weights"),
	)
	rows = append(rows, w.checkpointRows()...)
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

func (w *Welcome) footer() string {
	keys := []string{
		keyStyle.Render("enter") + dimStyle.Render(" choose a model"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	return strings.Join(keys, dimStyle.Render(" · ")) + "\n " +
		dimStyle.Render("go run ./cmd/inspect to step through a pass")
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
