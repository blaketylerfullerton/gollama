package tui

import (
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// The llama is drawn from half-block characters rather than ASCII punctuation
// because a terminal cell is twice as tall as it is wide: ▀ and ▄ are the only
// way to get something that reads as a curve instead of a staircase.
//
// Every line is one entry in the slice and is padded to llamaWidth, so lipgloss
// can put a panel beside it without the ragged right edge pulling the layout
// around. A single raw string would work too, but the source indentation ends up
// inside the literal and the width invariant becomes impossible to test.
const (
	llamaWidth  = 38
	llamaHeight = 21
)

// The animation is two independent motions over the same face-on drawing: the
// head nods, and the eyes blink.
//
// The nod is a whole cell rather than the half cell ▀/▄ would allow. Half a cell
// re-phases every glyph in the head — the same pixels, but packed into different
// characters — so the ears appear to change shape rather than move. Shifting by a
// full row keeps the glyphs byte-identical and the motion reads cleanly. The row
// the head gains is taken back out of the neck, which is flat colour for ten
// pixel rows either side of the seam, so the body and all four feet stay put.
//
// Index is the nod phase (0 up, 1 down) plus 2 when the eyes are shut.
var llamaFrames = [][]string{
	{ // head up, eyes open
		"            ▄▄▄▄▄▄███▄▄▄▄▄▄           ",
		"    ▄▄  ▄████████████████████▄▄▄▄▄▄   ",
		" ▄██████▀▀▀   ▀▀████▀▀     ▀▀███████  ",
		" ███████▄███     ███████     ███████  ",
		"  ▀█████▄▀▀    ▄██████▀   ▄▄▄█████▄   ",
		"   ████████▄▄██████████████████████   ",
		"   ██████████████▀█▀███████████████   ",
		"   ██████████████▄█▄███████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		" ▄▄████████████████████████████████▄▄ ",
		"██████████████████████████████████████",
		" ▀ ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ▀██████████████████████████████▀   ",
		"    ▀████████████████████████████▀    ",
		"     ▄██████████████████████████▄     ",
		"    █████▀▀▀█████████████▀▀▀▀▀███▀    ",
	},
	{ // head down, eyes open
		"                                      ",
		"            ▄▄▄▄▄▄███▄▄▄▄▄▄           ",
		"    ▄▄  ▄████████████████████▄▄▄▄▄▄   ",
		" ▄██████▀▀▀   ▀▀████▀▀     ▀▀███████  ",
		" ███████▄███     ███████     ███████  ",
		"  ▀█████▄▀▀    ▄██████▀   ▄▄▄█████▄   ",
		"   ████████▄▄██████████████████████   ",
		"   ██████████████▀█▀███████████████   ",
		"   ██████████████▄█▄███████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		" ▄▄████████████████████████████████▄▄ ",
		"██████████████████████████████████████",
		" ▀ ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ▀██████████████████████████████▀   ",
		"    ▀████████████████████████████▀    ",
		"     ▄██████████████████████████▄     ",
		"    █████▀▀▀█████████████▀▀▀▀▀███▀    ",
	},
	{ // head up, eyes shut
		"            ▄▄▄▄▄▄███▄▄▄▄▄▄           ",
		"    ▄▄  ▄████████████████████▄▄▄▄▄▄   ",
		" ▄██████████████████████████████████  ",
		" ███████████████████████████████████  ",
		"  ▀███████████████████████████████▄   ",
		"   ████████████████████████████████   ",
		"   ██████████████▀█▀███████████████   ",
		"   ██████████████▄█▄███████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		" ▄▄████████████████████████████████▄▄ ",
		"██████████████████████████████████████",
		" ▀ ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ▀██████████████████████████████▀   ",
		"    ▀████████████████████████████▀    ",
		"     ▄██████████████████████████▄     ",
		"    █████▀▀▀█████████████▀▀▀▀▀███▀    ",
	},
	{ // head down, eyes shut
		"                                      ",
		"            ▄▄▄▄▄▄███▄▄▄▄▄▄           ",
		"    ▄▄  ▄████████████████████▄▄▄▄▄▄   ",
		" ▄██████████████████████████████████  ",
		" ███████████████████████████████████  ",
		"  ▀███████████████████████████████▄   ",
		"   ████████████████████████████████   ",
		"   ██████████████▀█▀███████████████   ",
		"   ██████████████▄█▄███████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		" ▄▄████████████████████████████████▄▄ ",
		"██████████████████████████████████████",
		" ▀ ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ████████████████████████████████   ",
		"   ▀██████████████████████████████▀   ",
		"    ▀████████████████████████████▀    ",
		"     ▄██████████████████████████▄     ",
		"    █████▀▀▀█████████████▀▀▀▀▀███▀    ",
	},
}

const (
	// llamaInterval is the clock the whole animation runs off. It only has to be
	// short enough that the blink can be brief; the nod counts ticks to go slower.
	llamaInterval = 130 * time.Millisecond
	// nodTicks is how long the head holds at the top or the bottom, so a full nod
	// takes twice this. Slow enough to read as breathing rather than a twitch.
	nodTicks = 5
	// The eyes shut for blinkTicks out of every blinkPeriod. The period is
	// deliberately not a multiple of a full nod, so the blink lands somewhere
	// different each time round and the loop doesn't announce itself.
	blinkPeriod = 47
	blinkTicks  = 2
)

// llamaTickMsg advances the animation. It carries nothing: the model counts its
// own ticks, so a tick that arrives late or out of order costs a frame at worst.
type llamaTickMsg struct{}

// llamaTick schedules the next frame. Bubbletea has no repeating timer, so every
// tick has to ask for its successor.
func llamaTick() tea.Cmd {
	return tea.Tick(llamaInterval, func(time.Time) tea.Msg { return llamaTickMsg{} })
}

// llamaFrameAt picks the frame for a tick count.
func llamaFrameAt(tick int) []string {
	i := (tick / nodTicks) % 2
	if tick%blinkPeriod < blinkTicks {
		i += 2
	}
	return llamaFrames[i]
}

// llama renders the frame for tick as one styled block.
func llama(tick int) string {
	return llamaStyle.Render(strings.Join(llamaFrameAt(tick), "\n"))
}
