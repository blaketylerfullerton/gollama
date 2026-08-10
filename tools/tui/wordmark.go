package tui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// The name in large letters, for the top of the welcome screen.
//
// The letterforms are squared off rather than curved: at six rows a block
// character is the smallest mark available, so anything that tries to round a
// corner spends its only pixel doing it and comes out looking like a mistake.
// Flat tops on the o, a and m are a style rather than a compromise, and they
// keep the wordmark reading as one object beside the llama, which is drawn from
// the same character at the same weight.
//
// Lowercase letters leave their top two rows empty, so cap height and x-height
// come out of one grid.
var glyphs = map[rune][wordmarkRows]string{
	'G': {"███████", "██     ", "██     ", "██  ███", "██   ██", "███████"},
	'o': {"      ", "      ", "██████", "██  ██", "██  ██", "██████"},
	'L': {"██    ", "██    ", "██    ", "██    ", "██    ", "██████"},
	'l': {"██", "██", "██", "██", "██", "██"},
	'a': {"      ", "      ", "██████", "██  ██", "██████", "██  ██"},
	'm': {"            ", "            ", "████████████", "██   ██   ██", "██   ██   ██", "██   ██   ██"},
}

const wordmarkRows = 6

// wordmark renders name in those letters, one blank column between each, and
// reports how wide the result is. Letters it has no glyph for are skipped: the
// map covers the program's own name and nothing else, and a missing letter
// should leave a gap rather than stop the screen from drawing.
func wordmark(name string) (string, int) {
	var rows [wordmarkRows]string
	var n int
	for _, r := range name {
		g, ok := glyphs[r]
		if !ok {
			continue
		}
		for i := range rows {
			if n > 0 {
				rows[i] += " "
			}
			rows[i] += g[i]
		}
		n++
	}
	return wordmarkStyle.Render(strings.Join(rows[:], "\n")), lipgloss.Width(rows[0])
}
