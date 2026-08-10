package tui

import (
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/sysinfo"
)

// Start runs the whole splash flow: what this machine is, then what to run on
// it. The two are separate screens because they answer separate questions, and
// the second one's answer depends on having read the first — how much memory is
// free is the constraint the picker is doing arithmetic against.
//
// b or esc on the picker goes back to the specs, so the loop here is a real
// back button rather than a one-way sequence.
//
// It returns the chosen model and whether to go ahead at all; a false means the
// user quit and the caller should exit without loading anything.
func Start(checkpointDir string) (Model, bool, error) {
	// Detected once and handed to both screens. Detection shells out to sysctl
	// and vm_stat, and doing that again between two frames would be a visible
	// pause for numbers that can't have changed.
	sys := sysinfo.Detect()
	root := filepath.Dir(checkpointDir)

	for {
		w := NewWelcomeFor(sys, checkpointDir)
		if _, err := tea.NewProgram(w, tea.WithAltScreen()).Run(); err != nil {
			return Model{}, false, err
		}
		if w.Choice() != Run {
			return Model{}, false, nil
		}

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
	}
}
