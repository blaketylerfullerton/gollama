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
	// The wordmark takes the undiluted hue as ink rather than as a field: it's
	// large enough that reversing it would put a block of solid colour a
	// quarter the size of the screen behind two words.
	wordmarkStyle = amber.Fg(amber.Core)
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

	// The toolbar takes the same border as the panels — it's part of the same
	// frame — but a step dimmer. It says the same thing on every screen, so it
	// shouldn't compete for attention with the numbers above it.
	toolbarStyle = panelStyle.BorderForeground(amber.At(amber.Ash))
)

var (
	// The two speakers in the transcript get the same treatment as any other
	// label on screen: a step below the values they introduce. Colour is the
	// only thing that tells "you" from "model" apart at a glance, so the two
	// take opposite ends of the label range rather than sharing one.
	youStyle        = amber.Fg(amber.Bright).Bold(true)
	modelStyle      = amber.Fg(amber.Faint).Bold(true)
	modelReplyStyle = amber.Fg(amber.Live)
)

var (
	selectedStyle = amber.Fg(amber.Hot).Bold(true)

	// The label column in the memory panel is narrower than the welcome
	// screen's, since the panel beside it is what needs the room.
	memLabelStyle = amber.Fg(amber.Faint).Width(13)
)

// The memory panel is a fixed width — it's columns of numbers, and reflowing
// those gains nothing — so it keeps its size at every terminal width and the
// list beside it takes whatever is left. Below the two of them side by side
// they stack instead, which is what the welcome screen already does.
//
// These are content widths — what a line inside the panel may occupy. The
// rendered panel is 6 cells wider: two of padding on each side, and one of
// border. Every string in picker.go is written to fit one of them exactly, so
// changing a number here means re-checking the lines that go inside it.
const (
	listInnerWidth = 46 // the narrowest the list is worth drawing, not its width
	memInnerWidth  = 28
	panelChrome    = 6 // padding (4) + border (2)
)

var memPanelStyle = panelStyle.Width(memInnerWidth + 4)
