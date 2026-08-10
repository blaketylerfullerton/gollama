package tui

import (
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
)

// Every colour here is a level on the amber ramp, and the level is the only
// thing being chosen — see tools/amber for why. cmd/inspect draws from the same
// ramp at the same levels, so the two programs aren't just a matching palette:
// a dim label means the same thing on this screen as a dim attention weight
// does on that one.
var (
	// Reversed rather than coloured: the title is the brightest thing on the
	// screen by area, so it takes the hue as a field and gives up the text.
	titleStyle = lipgloss.NewStyle().Bold(true).
			Foreground(amber.At(amber.Void)).Background(amber.At(amber.Core)).
			Padding(0, 1)
	subtitleStyle = amber.Fg(amber.Faint).Italic(true)
	dimStyle      = amber.Fg(amber.Faint)
	labelStyle    = amber.Fg(amber.Faint).Width(11)
	valueStyle    = amber.Fg(amber.Live)
	// A section heading names the numbers under it; it isn't one of them, so it
	// sits a step below the values it introduces and takes the undiluted hue
	// plus bold instead of extra brightness. Pushing headings to the top of the
	// ramp would make the labels outshine the data on every screen.
	headingStyle = amber.Fg(amber.Core).Bold(true)
	warnStyle    = amber.Fg(amber.Peak).Bold(true)
	llamaStyle   = amber.Fg(amber.Core)
	keyStyle     = amber.Fg(amber.Bright).Bold(true)

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).BorderForeground(amber.At(amber.Ember)).
			Padding(0, 2)
)

var (
	selectedStyle = amber.Fg(amber.Hot).Bold(true)

	// The label column in the memory panel is narrower than the welcome
	// screen's, since the panel beside it is what needs the room.
	memLabelStyle = amber.Fg(amber.Faint).Width(13)
)

// The two top panels are fixed width so their contents can be laid out as
// columns of numbers rather than reflowed on every resize. Below the combined
// width they stack instead, which is what the welcome screen already does.
//
// These are content widths — what a line inside the panel may occupy. The
// rendered panel is 6 cells wider: two of padding on each side, and one of
// border. Every string in picker.go is written to fit one of them exactly, so
// changing a number here means re-checking the lines that go inside it.
const (
	listInnerWidth = 46
	memInnerWidth  = 28
	panelChrome    = 6 // padding (4) + border (2)

	// The two side by side, with a single cell between them.
	topWidth = listInnerWidth + panelChrome + 1 + memInnerWidth + panelChrome
)

var (
	listPanelStyle = panelStyle.Width(listInnerWidth + 4)
	memPanelStyle  = panelStyle.Width(memInnerWidth + 4)
)
