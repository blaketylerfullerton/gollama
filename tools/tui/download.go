package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
	"github.com/blaketylerfullerton/GoLlama/tools/hf"
	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// The screen between picking a model that isn't on disk yet and running it:
// rather than printing a huggingface-cli invocation and sending you to another
// terminal, this fetches it itself and shows the progress live. Package tui
// still doesn't import engine/model here — tools/hf only knows the handful of
// filenames a checkpoint is made of, the same way catalog.go only knows Arch.

// DownloadOutcome is how the download screen ended.
type DownloadOutcome int

const (
	// downloadRunning is the zero value, meaning it's still going — Outcome
	// only means anything once done has fired.
	downloadRunning DownloadOutcome = iota
	// DownloadDone means the checkpoint is on disk; go run it.
	DownloadDone
	// DownloadFailed means the fetch errored out; Err has why.
	DownloadFailed
	// DownloadBack means the user cancelled and wants the picker back.
	DownloadBack
	// DownloadQuit means the user cancelled and wants out of the program
	// entirely, the same q/ctrl+c every other screen answers to.
	DownloadQuit
)

type downloadProgressMsg hf.Progress
type downloadDoneMsg struct{}
type downloadErrMsg struct{ err error }

// Download is the bubbletea model for the fetch-in-progress screen.
type Download struct {
	model  Model
	events <-chan tea.Msg
	cancel context.CancelFunc

	outcome DownloadOutcome
	err     error
	prog    hf.Progress
	started time.Time

	w, h int
}

var _ tea.Model = (*Download)(nil)

// NewDownload starts fetching m's weights in the background and returns a
// screen that watches it happen. m.Repo is assumed non-empty: the only models
// that ever reach here are catalog entries missing their weights, and every
// one of those names a repo to fetch them from.
func NewDownload(m Model) *Download {
	ctx, cancel := context.WithCancel(context.Background())
	events := make(chan tea.Msg, 4)
	go runDownload(ctx, m, events)
	return &Download{model: m, events: events, cancel: cancel, started: time.Now(), w: 100, h: 32}
}

// runDownload does the fetch and turns its outcome into the one message the
// screen was still waiting on. Progress reports and that final message share
// one channel, closed once this returns, so waitForDownload never has to
// choose between two sources.
func runDownload(ctx context.Context, m Model, out chan<- tea.Msg) {
	defer close(out)
	err := hf.Download(ctx, m.Repo, m.Dir, func(p hf.Progress) {
		select {
		case out <- downloadProgressMsg(p):
		case <-ctx.Done():
		}
	})
	var final tea.Msg = downloadDoneMsg{}
	if err != nil {
		final = downloadErrMsg{err: err}
	}
	select {
	case out <- final:
	case <-ctx.Done():
	}
}

// Outcome reports how the screen ended. Valid once it has finished.
func (d *Download) Outcome() DownloadOutcome { return d.outcome }

// Err is why the download failed. Valid when Outcome is DownloadFailed.
func (d *Download) Err() error { return d.err }

// Model is what was being fetched, for the caller to hand back to the picker
// once it knows the fetch is done.
func (d *Download) Model() Model { return d.model }

func (d *Download) Init() tea.Cmd { return waitForDownload(d.events) }

// waitForDownload is the same one-message-at-a-time channel drain every other
// live screen in this package uses — see chat.go's waitForChat.
func waitForDownload(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return nil
		}
		return msg
	}
}

func (d *Download) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		d.w, d.h = msg.Width, msg.Height
	case downloadProgressMsg:
		d.prog = hf.Progress(msg)
		return d, waitForDownload(d.events)
	case downloadDoneMsg:
		d.outcome = DownloadDone
		return d, done
	case downloadErrMsg:
		d.outcome, d.err = DownloadFailed, msg.err
		return d, done
	case tea.KeyMsg:
		switch msg.String() {
		case "esc", "b", "backspace", "left":
			d.cancel()
			d.outcome = DownloadBack
			return d, done
		case "q", "ctrl+c":
			d.cancel()
			d.outcome = DownloadQuit
			return d, done
		}
	}
	return d, nil
}

func (d *Download) View() string {
	bar := d.bar()
	inner := max(d.w-2*screenMargin, minSpecsWidth)
	panel := panelStyle.Width(inner - 2).Render(d.body(inner - 6))
	body := lipgloss.JoinVertical(lipgloss.Left,
		header("GoLlama", "fetching "+d.model.Name, inner), "", panel)
	return screen(d.w, d.h, body, bar)
}

func (d *Download) body(width int) string {
	rows := []string{heading(d.model.Name), ""}

	if d.prog.FileCount == 0 {
		rows = append(rows, dimStyle.Render("contacting huggingface.co…"))
		return strings.Join(rows, "\n")
	}

	rows = append(rows,
		row("file", fmt.Sprintf("%s  (%d of %d)", d.prog.File, d.prog.FileIndex, d.prog.FileCount)),
		"",
		d.gauge(width),
		"",
		memRow("downloaded", valueStyle.Render(byteFraction(d.prog.Bytes, d.prog.Total))),
		memRow("speed", valueStyle.Render(d.speed())),
	)
	return strings.Join(rows, "\n")
}

// gauge is the same brighter-as-it-fills bar the picker draws for memory
// headroom, here for download progress instead — one visual language for
// "how much of this is done" everywhere GoLlama uses it.
func (d *Download) gauge(width int) string {
	if d.prog.Total <= 0 {
		return dimStyle.Render("size unknown — downloading anyway")
	}
	frac := float64(d.prog.Bytes) / float64(d.prog.Total)
	style := lipgloss.NewStyle().Foreground(amber.Of(frac))

	barWidth := min(max(width-8, 10), 40)
	filled := min(int(frac*float64(barWidth)+0.5), barWidth)
	bar := style.Render(strings.Repeat("█", filled)) +
		amber.NFg(amber.Rule).Render(strings.Repeat("░", barWidth-filled))
	return bar + style.Render(fmt.Sprintf(" %3.0f%%", frac*100))
}

// speed is bytes-per-second averaged over the whole download so far, plain
// and steady rather than an instantaneous rate that jitters every report.
func (d *Download) speed() string {
	elapsed := time.Since(d.started).Seconds()
	if elapsed <= 0 || d.prog.Bytes == 0 {
		return "—"
	}
	return sysinfo.Bytes(int64(float64(d.prog.Bytes)/elapsed)) + "/s"
}

func byteFraction(done, total int64) string {
	if total <= 0 {
		return sysinfo.Bytes(done)
	}
	return sysinfo.Bytes(done) + " / " + sysinfo.Bytes(total)
}

func (d *Download) bar() string {
	keys := []string{
		keyStyle.Render("esc") + dimStyle.Render(" cancel"),
		keyStyle.Render("q") + dimStyle.Render(" quit"),
	}
	left := strings.Join(keys, dimStyle.Render(" · "))
	if d.err != nil {
		left += "\n" + warnStyle.Render(d.err.Error())
	}
	return toolbar(d.w, left, "")
}
