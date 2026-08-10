package tui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Both screens are drawn to the same frame: a body that takes whatever room the
// terminal has, and a toolbar of keys pinned along the bottom of it. The frame
// is here rather than in each screen because the two have to agree — the
// welcome screen and the picker are the same window a keystroke apart, and keys
// that move between them read as a redraw rather than as a new screen.
//
// The frame fills the terminal. That isn't decoration: both screens run on the
// alternate screen, so anything they don't draw is blank terminal, and a layout
// that stops two thirds of the way down looks like it failed to finish rather
// than like it's done.

// screenMargin is the blank column kept down each side, so no border sits
// against the edge of the terminal.
const screenMargin = 1

// bodyRows is how many rows the body gets at terminal height h — everything but
// the blank line at the top and the toolbar at the bottom. The bar is measured
// rather than assumed: it grows a line when there's something to say beyond the
// keys, and the body has to give that line up rather than be clipped for it.
//
// The floor is for the degenerate case, a terminal too short to hold the frame
// at all. The body overflows there and screen() clips it; a negative height
// would instead collapse the panels into their own borders.
func bodyRows(h int, bar string) int {
	return max(h-1-lipgloss.Height(bar), 6)
}

// screen assembles the frame. The toolbar is pinned to the bottom of the
// terminal rather than left to float under the body, so the keys are in the
// same place on both screens whatever the body above them is doing.
func screen(w, h int, body, bar string) string {
	_ = w // the body and the bar are already sized to the width.

	lines := append([]string{""}, strings.Split(indent(body, screenMargin), "\n")...)
	tail := strings.Split(indent(bar, screenMargin), "\n")

	switch fill := h - len(lines) - len(tail); {
	case fill > 0:
		lines = append(lines, make([]string, fill)...)
	case fill < 0:
		// Too tall to fit. Clip the body rather than letting it push the
		// toolbar off the screen — losing the last row of a panel is
		// recoverable by resizing, losing the key that quits is not.
		lines = lines[:max(len(lines)+fill, 1)]
	}
	return strings.Join(append(lines, tail...), "\n")
}

// toolbar is the bar along the bottom: what the keys do on the left, and
// whatever else is worth saying on the right, pushed to the far end.
//
// It spans the terminal because a strip of keys that stops two thirds of the
// way across reads as one more leftover panel rather than as part of the frame.
// When there isn't room for both halves the right one goes: it's a suggestion,
// and the left half is how you get out of the program.
func toolbar(w int, left, right string) string {
	width := max(w-2*screenMargin, 24) - 2 // less its own border
	if gap := width - 4 - lipgloss.Width(left) - lipgloss.Width(right); gap >= 2 {
		left += strings.Repeat(" ", gap) + right
	}
	return toolbarStyle.Width(width).Render(left)
}

// header is a title with its subtitle beside it, sized to whatever it's sitting
// on top of. The subtitle is dropped rather than allowed to run past the panel
// below it, since a heading wider than its own box stops reading as a heading.
//
// This is the working screens' title: one row, because every row it takes is a
// row of model list it takes away. The splash screen uses hero instead.
func header(title, subtitle string, width int) string {
	t := titleStyle.Render(title)
	if lipgloss.Width(t)+1+lipgloss.Width(subtitle) <= width {
		return t + " " + subtitleStyle.Render(subtitle)
	}
	return t
}

// hero is the title in large letters with the subtitle on its own line
// underneath, for the screen that has the room to introduce the program rather
// than just label itself.
//
// It falls back to header when the letters won't fit across the width they're
// sitting on: a wordmark wrapped over two lines is worse than a small one.
func hero(title, subtitle string, width int) string {
	art, artWidth := wordmark(title)
	if artWidth > width {
		return header(title, subtitle, width)
	}
	return lipgloss.JoinVertical(lipgloss.Left, art, "", subtitleStyle.Render(subtitle))
}

// stretch renders s inside style, grown to occupy n rows when the content is
// shorter than that.
//
// It only ever grows. A panel set shorter than its contents clips them, and a
// box with a gap at the bottom is a better outcome than one whose last line has
// silently gone missing.
func stretch(style lipgloss.Style, n int, s string) string {
	if inner := n - 2; inner > lipgloss.Height(s) { // n includes the border
		style = style.Height(inner)
	}
	return style.Render(s)
}
