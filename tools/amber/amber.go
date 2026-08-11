// Package amber is the palette for every screen GoLlama draws.
//
// It has one hue. Everything on screen is amber at hue 38°, and the only thing
// that varies is how bright it is. That constraint is not decoration — it's the
// whole reason the palette is a package instead of a handful of colour literals
// next to the styles that use them.
//
// A transformer trace is full of numbers that are all the same kind of thing:
// an attention weight, a token probability, a fraction of memory in use. The
// obvious way to colour them is by category — blue for attention, green for
// probability, red when something's wrong — and it's the wrong way, because it
// spends the eye's most sensitive channel on a distinction the reader already
// knows (they can see which panel they're in) and leaves the distinction they
// can't make on their own — how big is this number, really — to be read digit
// by digit.
//
// So brightness carries magnitude, and nothing else does. Across the attention
// matrix, the logit-lens bars, the memory gauge, and the chrome around them, a
// brighter cell always means more: more weight, more probability, more of the
// machine consumed. Two bright cells in different panels are making the same
// claim. That is a property you get by having exactly one ramp and no escape
// hatch from it, which is what this package is.
package amber

import (
	"math"

	"github.com/charmbracelet/lipgloss"
)

// Ramp is the palette: ten steps of hue 38°, from near-black to near-white.
//
// The hue and saturation are fixed and the lightness climbs in even perceptual
// steps, so the sequence has no landmarks in it — no point where the colour
// changes character and invites being read as a category. Adjacent steps are
// far enough apart to tell apart side by side, which is what a heat matrix
// needs, and close enough that a gradient across several of them reads as a
// gradient rather than as banding.
//
// The ends are deliberately not pure black and pure white. Ramp[0] is a very
// dark brown that still reads as warm against a black terminal, so a
// zero-activation cell is visibly *part of the picture* rather than a hole in
// it; Ramp[9] is a cream rather than white, so peak activation still belongs to
// the same family as everything below it.
var Ramp = [10]lipgloss.Color{
	"#151B29", // 0  nothing here
	"#1F2C47", // 1
	"#2C4270", // 2
	"#385899", // 3
	"#4472CC", // 4
	"#4F8CFF", // 5  the hue, undiluted
	"#7CA6FF", // 6
	"#A6C2FF", // 7
	"#CFDEFF", // 8
	"#EFF4FF", // 9  as loud as it gets
}

// Named steps, for the places that pick a level rather than compute one.
//
// The names describe position on the ramp, not what the thing is for. A style
// asking for Faint is saying "this is low on the scale", and it has to mean
// that in the chrome for the same reason it means it in the matrix: a footer
// hint and a 0.03 attention weight are both things you should be able to skip
// past, and they should look equally skippable.
//
// The bottom of the ramp is deliberately not available to anything with words
// in it. Against a dark terminal Ramp[3] comes out around 3.4:1, under the
// 4.5:1 that body text needs, and Faint carries real prose — the paragraphs on
// the welcome screen, the model notes in the picker. So every level that holds
// text starts at 4, and 3 and below are left to data cells and rules, where a
// glyph that fades to almost nothing is the honest rendering of a number that
// is almost nothing. Data gets the whole ramp; text gets the top of it.
const (
	Void   = 0 // absent, masked, or off — and the ink on a reversed field
	Ash    = 1 // rules, the empty half of a gauge
	Ember  = 2 // borders, panel edges
	Faint  = 4 // labels, prose, headings, anything explanatory
	Core   = 5 // the undiluted hue: title fields, the llama
	Live   = 6 // values you're meant to actually read
	Bright = 7 // keys, the things you can press
	Hot    = 8 // the answer, the selected row
	Peak   = 9 // errors and warnings — the loudest thing available
)

// At returns the colour at a level, clamped to the ramp.
func At(level int) lipgloss.Color {
	switch {
	case level < 0:
		return Ramp[0]
	case level >= len(Ramp):
		return Ramp[len(Ramp)-1]
	}
	return Ramp[level]
}

// Fg is a style that is nothing but a foreground colour at a level.
func Fg(level int) lipgloss.Style {
	return lipgloss.NewStyle().Foreground(At(level))
}

// gamma bends the 0..1 input before it hits the ramp.
//
// Straight linear mapping wastes the palette on data like this. Attention rows
// sum to one across a dozen or more positions, so almost every cell lives below
// 0.1 and a linear map would render the entire matrix in the bottom two steps —
// technically faithful, and it shows you nothing. At 0.45 the low end gets most
// of the ramp: 0.02 and 0.10 land two steps apart instead of sharing one, while
// 0.5 and 1.0 still separate cleanly at the top.
const gamma = 0.45

// Of maps a strength in 0..1 onto the ramp. This is the function the whole
// palette exists for — anywhere the number is known, the colour should come
// from here rather than from a hand-picked level, so that "brighter means more"
// stays true by construction instead of by everyone remembering to honour it.
func Of(strength float64) lipgloss.Color {
	return At(LevelOf(strength))
}

// LevelOf is Of as a ramp index, for callers that want to compare two
// strengths' levels rather than colour something.
func LevelOf(strength float64) int {
	switch {
	case strength <= 0:
		return 0
	case strength >= 1:
		return len(Ramp) - 1
	}
	return int(math.Pow(strength, gamma)*float64(len(Ramp)-1) + 0.5)
}
