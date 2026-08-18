package tui

import (
	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
)

// Almost everything here is on the furniture track — see tools/amber. The hue
// appears exactly three times on this screen: the title field, the wordmark,
// and the keys you can press. That scarcity is the point. Everything else is
// grey, which is what lets those three read as emphasis and what leaves the
// data in cmd/inspect as the only saturated thing in its own frame.
var (
	// Reversed rather than coloured: the title is the brightest thing on the
	// screen by area, so it takes the hue as a field and gives up the text.
	titleStyle = lipgloss.NewStyle().Bold(true).
			Foreground(amber.N(0)).Background(amber.At(amber.Accent)).
			Padding(0, 1)
	// The wordmark takes the undiluted hue as ink rather than as a field: it's
	// large enough that reversing it would put a block of solid colour a
	// quarter the size of the screen behind two words.
	wordmarkStyle = amber.Fg(amber.Accent)
	subtitleStyle = amber.NFg(amber.Muted).Italic(true)
	dimStyle      = amber.NFg(amber.Muted)
	labelStyle    = amber.NFg(amber.Muted).Width(11)
	valueStyle    = amber.NFg(amber.Body)
	// A heading outranks the values under it now, which is the opposite of how
	// this read before. On the old single ramp a bright heading would have
	// claimed to be a large number, so headings were held below their own
	// content and bold was left to compensate. Off the data track that worry is
	// gone: nothing about a grey heading suggests a magnitude, so it can simply
	// be the loudest text in its section, which is what it is.
	headingStyle = amber.NFg(amber.Strong).Bold(true)
	warnStyle    = lipgloss.NewStyle().Foreground(amber.Alert).Bold(true)
	// Keys carry the hue and their descriptions don't, so a toolbar reads as a
	// row of pressable things with grey annotations rather than as an
	// undifferentiated strip of text.
	keyStyle = amber.Fg(amber.Accent).Bold(true)

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).BorderForeground(amber.N(amber.Edge)).
			Padding(0, 2)

	// headerRowStyle names the columns above a table (the inspect screen's
	// logit-lens/attention/attribution rows) — off the data track like any
	// other heading, so it doesn't compete with the magnitudes underneath it.
	headerRowStyle = amber.NFg(amber.Body).Bold(true)
	// tabStyle/activeTabStyle are the inspect screen's view switcher — the
	// same reversed-field treatment titleStyle/selectedStyle already use for
	// "the thing you're on", just narrower.
	tabStyle       = amber.NFg(amber.Muted)
	activeTabStyle = lipgloss.NewStyle().Bold(true).
			Foreground(amber.N(0)).Background(amber.At(amber.Accent))

	// ruleStyle draws a divider a title badge sits on top of — see titleRule.
	// A step below the panel border, the same weight the toolbar's own frame
	// uses, so the line reads as chrome rather than as content of its own.
	ruleStyle = amber.NFg(amber.Rule)

	// The toolbar takes the same border as the panels — it's part of the same
	// frame — but a step dimmer. It says the same thing on every screen, so it
	// shouldn't compete for attention with the numbers above it.
	toolbarStyle = panelStyle.BorderForeground(amber.N(amber.Rule))
)

var (
	// Colour is what tells "you" from "model" apart at a glance, and now the
	// two speakers can differ by hue instead of by two steps of brightness:
	// your turn is the accent, the model's is grey. That also matches who is
	// acting — the accent marks the things you do everywhere else on screen.
	youStyle        = amber.Fg(amber.Accent).Bold(true)
	modelStyle      = amber.NFg(amber.Muted).Bold(true)
	modelReplyStyle = amber.NFg(amber.Body)
)

var (
	// The selected row is a field rather than bright ink. A row is wide enough
	// to carry a background without shouting, and reversing it survives being
	// surrounded by other text in a way that one more step of brightness on a
	// list of already-bright rows does not.
	selectedStyle = lipgloss.NewStyle().Bold(true).
			Foreground(amber.N(0)).Background(amber.At(amber.Accent))

	// The label column in the memory panel is narrower than the welcome
	// screen's, since the panel beside it is what needs the room.
	memLabelStyle = amber.NFg(amber.Muted).Width(13)
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
