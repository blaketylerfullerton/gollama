package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// b, backspace and esc are the three ways back to the menu; a typo in any of
// them leaves someone stuck reading prose with no way out but ctrl+c.
func TestAboutBackKeys(t *testing.T) {
	for _, k := range []string{"b", "backspace", "esc"} {
		a := NewAbout()
		_, cmd := a.Update(key(k))
		if cmd == nil {
			t.Errorf("%q did not end the page", k)
			continue
		}
		if a.Outcome() != AboutBack {
			t.Errorf("%q gave outcome %v, want AboutBack", k, a.Outcome())
		}
	}
}

func TestAboutQuitKeys(t *testing.T) {
	for _, k := range []string{"q", "ctrl+c"} {
		a := NewAbout()
		_, cmd := a.Update(key(k))
		if cmd == nil {
			t.Errorf("%q did not end the page", k)
			continue
		}
		if a.Outcome() != AboutQuit {
			t.Errorf("%q gave outcome %v, want AboutQuit", k, a.Outcome())
		}
	}
}

// The page exists to say what this program is; whatever else changes about
// its layout, that text has to actually be on screen.
func TestAboutShowsItsProse(t *testing.T) {
	a := NewAbout()
	a.Update(tea.WindowSizeMsg{Width: 100, Height: 34})
	view := a.View()
	for _, want := range []string{"GoLlama", "scalar Go", "cmd/inspect"} {
		if !strings.Contains(view, want) {
			t.Errorf("view is missing %q", want)
		}
	}
}

// The viewport has no word-wrap of its own — it only clips a line to its
// width — so the prose has to be wrapped before it reaches the viewport, or
// everything past the first screenful of a paragraph is silently dropped
// rather than scrolled to. This is the regression the panel-arithmetic bug
// caused: content wrapped 2 cells too wide for the panel it sat inside.
func TestAboutWrapsRatherThanClipping(t *testing.T) {
	a := NewAbout()
	// Tall enough that every wrapped line fits without scrolling — this is
	// checking that the content exists at all, not that scrolling works.
	a.Update(tea.WindowSizeMsg{Width: 90, Height: 60})
	// The last word of the last paragraph — if wrapping silently clips a raw
	// long line instead of reflowing it, text this far into a paragraph never
	// renders at all. (Not asserting a longer phrase: a legitimate wrap can
	// land a newline anywhere between two words.)
	if !strings.Contains(a.View(), "legible") {
		t.Error("the tail of the prose never reached the screen — content is being clipped, not wrapped")
	}
}

// Same requirement as every other screen: whatever the terminal size, no line
// may run past it.
func TestAboutFitsTheTerminal(t *testing.T) {
	for _, size := range []tea.WindowSizeMsg{
		{Width: 60, Height: 20},
		{Width: 100, Height: 34},
		{Width: 160, Height: 45},
	} {
		a := NewAbout()
		a.Update(size)
		lines := strings.Split(a.View(), "\n")
		if len(lines) != size.Height {
			t.Errorf("at %dx%d: %d rows, want %d", size.Width, size.Height, len(lines), size.Height)
		}
		for _, line := range lines {
			if got := lipgloss.Width(line); got > size.Width {
				t.Errorf("at %dx%d: a %d-cell line: %q", size.Width, size.Height, got, line)
			}
		}
	}
}
