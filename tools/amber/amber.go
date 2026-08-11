// Package amber is the palette for every screen GoLlama draws.
//
// It has two tracks, and which one a thing belongs to is decided by a single
// question: is this a number, or is it furniture?
//
// A transformer trace is full of numbers that are all the same kind of thing:
// an attention weight, a token probability, a fraction of memory in use. The
// obvious way to colour them is by category — one hue for attention, another
// for probability, red when something's wrong — and it's the wrong way, because
// it spends the eye's most sensitive channel on a distinction the reader
// already knows (they can see which panel they're in) and leaves the
// distinction they can't make on their own — how big is this number, really —
// to be read digit by digit.
//
// So for data, brightness carries magnitude and nothing else does. Across the
// attention matrix, the logit-lens bars, and the memory gauge, a brighter cell
// always means more: more weight, more probability, more of the machine
// consumed. Two bright cells in different panels are making the same claim.
// That is a property you get by having exactly one ramp and no escape hatch
// from it, and Ramp is that ramp.
//
// The furniture is the other track, and it used to be on the first one. That
// was a mistake, and it's worth naming because the reasoning that produced it
// sounds right. If brightness means magnitude, the thinking went, then the
// chrome should be honest about its own level too — so borders, labels, and
// headings all got assigned a step on the same ramp. But a panel border is not
// a magnitude. Nothing is encoded by colouring it. All that rule accomplished
// was to put every word on screen inside the four steps in the middle of the
// ramp that are bright enough to read and dim enough not to be the answer, and
// four adjacent steps is not enough range to build a hierarchy out of. The
// screens came out flat, and bold had to do the work that value should have
// been doing.
//
// So the furniture moved to Neutral: the same ten-step construction, no
// saturation. Now the chrome recedes, prose can go to the top of the neutral
// track where it's properly legible, and the hue is scarce — it appears on the
// data and on the few things you can press, which is what makes it read as
// emphasis at all. Emphasis is relative. When everything was amber, nothing was.
//
// The one deliberate exception is Alert. An error is not a magnitude either,
// but unlike a border it genuinely is a category, it is rare, and it is the one
// place where a reader benefits from recognising the state before they've read
// the words. It gets a hue of its own and nothing else does.
package amber

import (
	"math"

	"github.com/charmbracelet/lipgloss"
)

// Ramp is the data track: ten steps of amber at hue 30°, near-black to near-white.
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
// it; Ramp[9] is a cream rather than white, so peak activation still
// belongs to the same family as everything below it.
var Ramp = [10]lipgloss.Color{
	"#2B1500", // 0  nothing here
	"#452300", // 1
	"#663400", // 2
	"#874500", // 3
	"#B35C00", // 4
	"#E87A0D", // 5  the hue, undiluted
	"#FF9E47", // 6
	"#FFBC80", // 7
	"#FFD9B5", // 8
	"#FFF0E0", // 9  as loud as it gets
}

// Named steps on the data track, for the places that pick a level rather than
// compute one.
//
// The names describe position on the ramp, not what the thing is for. A style
// asking for Core is saying "this is the middle of the scale", and it means the
// same thing here as it does in the matrix.
//
// The bottom of the ramp is not available to anything with words in it. Against
// a dark terminal Ramp[3] comes out at 2.9:1, under the 4.5:1 that body text
// needs. That constraint used to squeeze the whole interface, back when
// the chrome lived on this ramp; now that words are Neutral's problem, it binds
// on almost nothing, and the low steps are free to be what they should always
// have been — data cells and rules, where a glyph that fades to almost nothing
// is the honest rendering of a number that is almost nothing.
const (
	Void   = 0 // absent, masked, or off — and the ink on a reversed field
	Ash    = 1 // the empty half of a gauge
	Ember  = 2 // the dim end of a heat cell
	Faint  = 4 // low but present
	Core   = 5 // the undiluted hue: accents, title fields, the llama
	Live   = 6 // a value worth noticing
	Bright = 7 // the selected row's field
	Hot    = 8 // the answer
	Peak   = 9 // as far up as the data goes
)

// Neutral is the furniture track: the same ten-step construction at the same
// lightnesses, with the saturation taken out.
//
// It is not quite a straight grey, and the deviation is deliberate in both
// directions. The low steps are pulled very slightly cool, because that is the
// half of the track where the risk lives: a warm dark grey sitting next to the
// amber ramp reads as a *dimmer amber* — as data that has faded — and the whole
// point of the split is that furniture is a different kind of thing, not a
// weaker version of the same thing. Cool greys sit across the wheel from the
// hue, so a border reads as absent of colour rather than as low on the ramp.
//
// The top steps drift back warm, ending in a cream rather than a white. Up
// there the chroma is far too small to be seen as a hue at all — Neutral[9] is
// simply "white" to anyone reading it — so it costs nothing, and a page of text
// in a faintly warm white is easier to sit in front of than one in a clinical
// blue-white. Cool where it has to be, warm where it's free to be.
//
// The contrast floor that Ramp mostly escaped lands here instead, and harder,
// because this is where all the words are. Neutral[5] is about 4.2:1 on black —
// close, but under — so text starts at Muted and everything below it is
// structure: rules, borders, the parts of the frame you should be able to look
// past. That is the whole rule. Above the floor, prose; below it, furniture.
var Neutral = [10]lipgloss.Color{
	"#0E0E10", // 0  the darkest thing that isn't the terminal itself
	"#1A1B1D", // 1
	"#282A2D", // 2  rules, gauge tracks
	"#393B3F", // 3  borders
	"#545559", // 4
	"#6E7075", // 5  the contrast floor sits just above here
	"#919196", // 6  labels
	"#B4B3B2", // 7  secondary text
	"#D8D6D2", // 8  body text
	"#FAF9F5", // 9  headings, and the ink on a reversed field
}

// Named steps on the furniture track.
//
// Unlike the data names, these say what the thing is for rather than where it
// sits, because that's the actual decision a style is making. There is no
// magnitude being encoded — a label is not a smaller number than a heading — so
// naming them by level would imply a scale that isn't there.
const (
	Edge   = 3 // panel borders: the frame, visible but never read
	Rule   = 2 // dividers, gauge tracks, the toolbar — a step below the frame
	Muted  = 6 // labels, footer hints, anything you may skip
	Body   = 8 // prose and values: the text you are here to read
	Strong = 9 // headings, and the ink on a reversed field
)

// Accent is the one saturated thing in the chrome: keys, the wordmark, the
// selected row. It is deliberately scarce — it marks what you can act on, and
// it can only do that as long as it stays rare.
const Accent = Core

// Alert is the single exception to the two tracks — see the package comment.
// Errors are a state rather than a magnitude, and rare enough that giving them
// a hue costs nothing and buys recognition before the words are read.
//
// It is a rose rather than the obvious red, and that is forced by the accent.
// A plain error red lands around 5°, which is only about 25° from the amber at
// 30° — near enough that a warning and a keybinding read as the same family at
// a glance, and much nearer than that for the ~8% of men with a red-green
// deficiency, who would be left distinguishing them by brightness alone. Since
// the whole justification for Alert having a hue is that the hue is recognised
// before the words are, a hue that isn't reliably recognised is worse than none
// at all. Swinging to 349° buys real separation while still reading as alarm,
// which red's neighbours on the other side — going yellow — would not.
const Alert = lipgloss.Color("#FF4D6D")

// N returns the furniture colour at a level, clamped.
func N(level int) lipgloss.Color {
	switch {
	case level < 0:
		return Neutral[0]
	case level >= len(Neutral):
		return Neutral[len(Neutral)-1]
	}
	return Neutral[level]
}

// NFg is a style that is nothing but a furniture colour at a level.
func NFg(level int) lipgloss.Style {
	return lipgloss.NewStyle().Foreground(N(level))
}

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
