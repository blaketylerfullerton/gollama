package tui

import (
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// Start runs the whole splash flow: a start menu, then what to run. The menu's
// two rows lead to separate screens because they answer separate questions —
// one is "what should I run", which is the picker doing memory arithmetic
// against this machine; the other is "what even is this", which is prose. Both
// return to the menu rather than exiting through it, so backing out of either
// is a real back button and not a restart.
//
// It returns the chosen model and whether to go ahead at all; a false means the
// user quit and the caller should exit without loading anything.
func Start(checkpointDir string) (Model, bool, error) {
	// Detected once and handed to every screen. Detection shells out to sysctl
	// and vm_stat, and doing that again between frames would be a visible
	// pause for numbers that can't have changed.
	sys := sysinfo.Detect()
	root := filepath.Dir(checkpointDir)

	for {
		w := NewWelcomeFor(sys, checkpointDir)
		if _, err := tea.NewProgram(w, tea.WithAltScreen()).Run(); err != nil {
			return Model{}, false, err
		}

		switch w.Choice() {
		case ShowAbout:
			a := NewAbout()
			if _, err := tea.NewProgram(a, tea.WithAltScreen()).Run(); err != nil {
				return Model{}, false, err
			}
			if a.Outcome() == AboutQuit {
				return Model{}, false, nil
			}
			continue // AboutBack: reopen the menu

		case Run:
			p := NewPicker(root, sys)
			if _, err := tea.NewProgram(p, tea.WithAltScreen()).Run(); err != nil {
				return Model{}, false, err
			}
			switch p.Outcome() {
			case Selected:
				return p.Selection(), true, nil
			case Back:
				continue
			default:
				return Model{}, false, nil
			}

		default: // Quit
			return Model{}, false, nil
		}
	}
}
