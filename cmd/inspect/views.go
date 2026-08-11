package main

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

// This screen is where the two tracks earn their keep — see tools/amber. The
// attention matrix and the logit-lens bars are coloured by calling amber.Of on
// the number itself, and everything around them is grey, so the only saturated
// thing in the frame is the data. That used to be untrue: the chrome was on the
// same ramp, which meant a mid-level border and a mid-level attention weight
// were the same colour and the eye had no way to tell the reading from the
// furniture except by position.
var (
	titleStyle = lipgloss.NewStyle().Bold(true).Foreground(amber.N(0)).Background(amber.At(amber.Accent))
	dimStyle   = amber.NFg(amber.Muted)
	keyStyle   = amber.Fg(amber.Accent).Bold(true)
	hotStyle   = amber.NFg(amber.Strong).Bold(true)
	selStyle   = lipgloss.NewStyle().Bold(true).Foreground(amber.N(0)).Background(amber.At(amber.Accent))
	// A column heading names the numbers under it and is not one of them, so it
	// goes to the top of the furniture track. On the old shared ramp this had
	// to be held down at a low level to avoid outshining the data; off the data
	// track there is nothing to outshine, because grey makes no claim about
	// magnitude however bright it gets.
	headerRowStyle = amber.NFg(amber.Body).Bold(true)
	tabStyle       = amber.NFg(amber.Muted)
	activeTabStyle = lipgloss.NewStyle().Bold(true).Foreground(amber.N(0)).Background(amber.At(amber.Accent))
	errStyle       = lipgloss.NewStyle().Foreground(amber.Alert).Bold(true)
)

// --- logit lens -------------------------------------------------------------

// lensView shows what the model would predict at each depth. Reading it top to
// bottom is watching an answer get found: early layers guess generic function
// words, some middle layer commits to the actual token, and the last few sharpen
// or hedge it.
func (a *app) lensView() string {
	s := a.active()
	if s == nil {
		return ""
	}
	if len(s.lens) == 0 {
		return dimStyle.Render("  No logit-lens events in this trace.\n" +
			"  They're only recorded when a trace writer is attached, since each one\n" +
			"  costs an extra LM head projection.")
	}

	var b strings.Builder
	b.WriteString(headerRowStyle.Render(
		fmt.Sprintf("  %-7s %-16s %7s  %s", "layer", "prediction", "prob", "")) + "\n")

	// Find where the final answer first takes the lead, to mark it.
	firstWin := -1
	if final := a.finalPrediction(); final != nil {
		for _, e := range s.lens {
			if len(e.Top) > 0 && e.Top[0].ID == final.ID {
				firstWin = e.Layer
				break
			}
		}
	}

	lo, hi := window(len(s.lens), a.layer, a.bodyHeight()-1)
	for _, e := range s.lens[lo:hi] {
		if len(e.Top) == 0 {
			continue
		}
		top := e.Top[0]

		label := fmt.Sprintf("%d", e.Layer)
		if e.Layer >= s.tr.Header.Config.NLayer {
			label = "out" // the model's real output, not an intermediate read
		}

		row := fmt.Sprintf("  %-7s %-16s %6.1f%%  %s",
			label, truncate(fmt.Sprintf("%q", top.Text), 16), top.Prob*100, bar(top.Prob, 34))

		switch {
		case e.Layer == a.layer:
			b.WriteString(selStyle.Render(row))
		case e.Layer == firstWin:
			b.WriteString(hotStyle.Render(row))
		default:
			b.WriteString(row)
		}
		b.WriteString("\n")
	}

	if firstWin >= 0 && firstWin < s.tr.Header.Config.NLayer {
		b.WriteString("\n" + dimStyle.Render(fmt.Sprintf(
			"  the final answer first leads at layer %d of %d (highlighted)",
			firstWin, s.tr.Header.Config.NLayer)))
	}
	return b.String()
}

// --- attention --------------------------------------------------------------

func (a *app) attentionView() string {
	s := a.active()
	if s == nil {
		return ""
	}
	e, ok := s.tr.LayerEvent(a.layer, trace.KindAttention, a.head)
	if !ok || len(e.Weights) == 0 {
		return dimStyle.Render(fmt.Sprintf(
			"  No attention weights for layer %d head %d.\n"+
				"  Long sequences are skipped to keep trace files small.", a.layer, a.head))
	}

	toks := s.tr.Header.Tokens
	label := func(i int) string {
		if i < len(toks) {
			return truncate(sanitize(toks[i].Text), 8)
		}
		return fmt.Sprintf("t%d", i)
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("  layer %s  head %s   %s\n\n",
		keyStyle.Render(fmt.Sprint(a.layer)), keyStyle.Render(fmt.Sprint(a.head)),
		dimStyle.Render("each row attends across the columns")))

	// Column headings.
	b.WriteString(headerRowStyle.Render(fmt.Sprintf("  %-9s", "")))
	for j := range e.Weights {
		b.WriteString(headerRowStyle.Render(fmt.Sprintf("%9s", label(j))))
	}
	b.WriteString("\n")

	for i, row := range e.Weights {
		b.WriteString(fmt.Sprintf("  %-9s", label(i)))
		for j := range e.Weights {
			if j > i {
				// Never computed, not zeroed — the mask means these scores
				// don't exist. Drawn at the bottom of the ramp, below anything
				// a real weight can reach: on a scale where brightness is
				// magnitude, the absence of a number can't be allowed to
				// outshine a small one. The glyph is what says "masked"; the
				// darkness only says "nothing to see".
				b.WriteString(amber.Fg(amber.Ash).Render(fmt.Sprintf("%9s", "·")))
				continue
			}
			w := row[j]
			b.WriteString(heat(w).Render(fmt.Sprintf("%9.3f", w)))
		}
		b.WriteString("\n")
	}

	b.WriteString("\n" + dimStyle.Render("  · = masked. Each row sums to 1."))
	if len(e.Weights) > 1 {
		var sink float64
		for i := 1; i < len(e.Weights); i++ {
			sink += float64(e.Weights[i][0])
		}
		sink /= float64(len(e.Weights) - 1)
		b.WriteString(dimStyle.Render(fmt.Sprintf(
			"\n  token 0 absorbs %.0f%% of later positions' attention on average", sink*100)))
	}
	return b.String()
}

// heat colours a weight by magnitude, so a pattern is visible before any number
// is read.
//
// It was a six-colour rainbow — grey, blue, green, yellow, red — which had the
// flaw every rainbow scale has: the steps aren't ordered by anything the eye
// does automatically, so you can see that two cells differ without being able
// to see which one is larger, and reading the matrix meant consulting a legend
// you had to hold in your head. On the ramp, larger is brighter, and a row's
// shape falls out of it at a glance.
func heat(w float32) lipgloss.Style {
	return lipgloss.NewStyle().Foreground(amber.Of(float64(w)))
}

// --- stages -----------------------------------------------------------------

func (a *app) stagesView() string {
	s := a.active()
	if s == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("  layer %s\n\n", keyStyle.Render(fmt.Sprint(a.layer))))

	events := s.layers[a.layer]
	if len(events) == 0 {
		return b.String() + dimStyle.Render("  No events recorded for this layer.")
	}

	for _, e := range events {
		switch e.Kind {
		case trace.KindStage:
			b.WriteString(fmt.Sprintf("  %-24s %s  %s\n",
				e.Name,
				keyStyle.Render(fmt.Sprintf("‖x‖ %10.4f", e.MeanNorm)),
				dimStyle.Render(fmt.Sprintf("(%d tokens x %d dims)", e.Tokens, e.Dims))))
			if len(e.Preview) > 0 {
				b.WriteString(dimStyle.Render("    " + floats(e.Preview, 8) + "\n"))
			}
		case trace.KindRotary:
			if e.Head != a.head {
				continue
			}
			b.WriteString(fmt.Sprintf("  %-24s %s\n",
				fmt.Sprintf("rotary q (head %d)", e.Head),
				dimStyle.Render(fmt.Sprintf("‖v‖ %.4f → %.4f, cos %.4f",
					e.NormIn, e.NormOut, e.CosSim))))
			if e.NormIn > 0 {
				// Rotation is length-preserving; this is the check.
				drift := (e.NormOut - e.NormIn) / e.NormIn
				b.WriteString(dimStyle.Render(fmt.Sprintf(
					"    length drift %.2e — rotation only turns the vector\n", drift)))
			}
		case trace.KindNote:
			b.WriteString(dimStyle.Render("  note: " + e.Text + "\n"))
		}
	}
	return b.String()
}

// --- small helpers ----------------------------------------------------------

// bar draws a fraction as a filled row. The fill is coloured by the same
// fraction it's showing, so a confident prediction is both longer and brighter
// than a hedged one — the two encodings agree, and the column of bars can be
// skimmed for brightness alone.
func bar(frac float64, width int) string {
	n := int(frac * float64(width))
	n = max(0, min(width, n))
	return lipgloss.NewStyle().Foreground(amber.Of(frac)).Render(strings.Repeat("█", n)) +
		amber.NFg(amber.Rule).Render(strings.Repeat("░", width-n))
}

func floats(v []float32, n int) string {
	if n > len(v) {
		n = len(v)
	}
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = fmt.Sprintf("%+.4f", v[i])
	}
	s := strings.Join(parts, " ")
	if len(v) > n {
		s += fmt.Sprintf(" … %d dims", len(v))
	}
	return s
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n])
}

// sanitize makes whitespace visible, since a token's leading space is
// significant and invisible otherwise.
func sanitize(s string) string {
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return strings.ReplaceAll(s, " ", "_")
}
