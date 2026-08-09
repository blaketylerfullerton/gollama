package tui

import "github.com/charmbracelet/lipgloss"

// The palette is the one cmd/inspect already uses, so the two programs look like
// they came from the same place: 62 for chrome, 245 for anything secondary, 81
// for values you're meant to read, 212 for the one thing that matters most.
var (
	titleStyle = lipgloss.NewStyle().Bold(true).
			Foreground(lipgloss.Color("231")).Background(lipgloss.Color("62")).
			Padding(0, 1)
	subtitleStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("245")).Italic(true)
	dimStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	labelStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("245")).Width(11)
	valueStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("81"))
	hotStyle      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("212"))
	warnStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	llamaStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("180"))
	keyStyle      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("231"))

	panelStyle = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("240")).
			Padding(0, 2)
)
