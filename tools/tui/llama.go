package tui

import "strings"

// The llama is drawn from half-block characters rather than ASCII punctuation
// because a terminal cell is twice as tall as it is wide: ▀ and ▄ are the only
// way to get something that reads as a curve instead of a staircase.
//
// Every line is padded to llamaWidth so lipgloss can put a panel beside it
// without the ragged right edge pulling the layout around.
const llamaWidth = 30

// Drawn in profile: it's the silhouette that reads as a llama rather than as a
// generic animal face — the long neck is the whole tell.
var llamaArt = []string{
	`                   ▄▄  ▄▄     `,
	`                   ██  ██     `,
	`                  ▀███████▄▄  `,
	`                  █ o  ████   `,
	`                  ▀██▄▄▄███   `,
	`                   █████▀     `,
	`                   █████      `,
	`                  █████       `,
	`        ▄▄▄▄▄▄▄▄▄█████        `,
	` ▄▄███████████████████        `,
	`  ████████████████████        `,
	`  ████████████████████        `,
	`   ▀██▀▀██▀    ▀██▀▀██▀       `,
	`    ██  ██      ██  ██        `,
	`    ██  ██      ██  ██        `,
	`    ▀▀  ▀▀      ▀▀  ▀▀        `,
}

// llama renders the art as one styled block.
func llama() string {
	return llamaStyle.Render(strings.Join(llamaArt, "\n"))
}
