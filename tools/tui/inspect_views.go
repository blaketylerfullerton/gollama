package tui

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

// This is where the two tracks earn their keep — see tools/amber. The
// attention bars and the logit-lens bars are coloured by calling amber.Of on
// the number itself, and everything around them is grey, so the only saturated
// thing in the frame is the data. Chrome shared with the rest of the package —
// titleStyle, dimStyle, keyStyle, warnStyle, headingStyle, selectedStyle,
// headerRowStyle, tabStyle, activeTabStyle — lives in style.go rather than
// being redefined here.

// --- logit lens -------------------------------------------------------------

// lensView shows what the model would predict at each depth. Reading it top to
// bottom is watching an answer get found: early layers guess generic function
// words, some middle layer commits to the actual token, and the last few sharpen
// or hedge it.
func (a *Inspect) lensView() string {
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
		fmt.Sprintf("  %-5s %-14s %7s %6s %6s  %s",
			"layer", "prediction", "prob", "rank", "H", "")) + "\n")

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

		// Where the answer stands at this depth, and how undecided the model is
		// overall. The two disagree often enough to be worth showing together: a
		// layer can have the right token in front and still be near-uniform
		// behind it, which reads as confidence from the top row alone.
		rank, ent := "—", "—"
		if e.TargetRank > 0 {
			rank = fmt.Sprintf("#%d", e.TargetRank)
		}
		if e.Entropy > 0 {
			ent = fmt.Sprintf("%.2f", e.Entropy)
		}

		row := fmt.Sprintf("  %-5s %-14s %6.1f%% %6s %6s  %s",
			label, truncate(fmt.Sprintf("%q", top.Text), 14), top.Prob*100,
			rank, ent, bar(top.Prob, 24))

		switch {
		case e.Layer == a.layer:
			b.WriteString(selectedStyle.Render(row))
		case e.Layer == firstWin:
			b.WriteString(headingStyle.Render(row))
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
//
// This used to be one full N×N matrix for a single (layer, head) — every
// query row at once, columns coloured by magnitude. Readable for a handful of
// tokens, but it doesn't scale: past a short prompt it's a wall of numbers,
// and there was no way to follow one token's story rather than the whole
// grid's. Two focused views replace it, both centred on a single query token
// (a.attnQuery, cycled with tab/shift+tab — see handleKey):
//
//   - the default, attentionBarsView: that token's weights over everything at
//     or before it, one layer/head at a time, as a bar chart — the same shape
//     as attentionExample() on the welcome screen, live instead of a mockup.
//   - attentionTraceView (toggled with "t"): the same token, but walking
//     every layer at the selected head, showing which earlier token it
//     attended to the most at each depth — the attention analogue of what
//     lensView already does for predictions.

// attentionView dispatches to whichever of the two the screen is currently
// showing; both need the same (layer, head) lookup and token-label function.
func (a *Inspect) attentionView() string {
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
			return truncate(sanitize(toks[i].Text), 12)
		}
		return fmt.Sprintf("t%d", i)
	}
	// Clamped rather than trusted: attnQuery is only reclamped on the key that
	// moves it (see handleKey), the same lag every other cursor in this screen
	// tolerates between steps of different lengths.
	q := min(a.attnQuery, len(e.Weights)-1)

	if a.attnTrace {
		return a.attentionTraceView(s, q, label)
	}
	return a.attentionBarsView(e, q, label)
}

// attentionBarsView shows one query token's weights over every token at or
// before it — a single row out of what used to be the whole matrix, wide
// enough to read at a glance instead of squinting at a grid.
func (a *Inspect) attentionBarsView(e trace.Event, q int, label func(int) string) string {
	row := e.Weights[q][:q+1]

	var b strings.Builder
	b.WriteString(fmt.Sprintf("  layer %s  head %s   query %s  %s\n\n",
		keyStyle.Render(fmt.Sprint(a.layer)), keyStyle.Render(fmt.Sprint(a.head)),
		headingStyle.Render(truncate(fmt.Sprintf("%q", label(q)), 14)),
		dimStyle.Render(fmt.Sprintf("(token %d of %d)", q, len(e.Weights)-1))))

	// Anchored on the query token itself (the last entry) rather than 0, so a
	// long prefix scrolls to show what's nearest it first.
	lo, hi := window(len(row), len(row)-1, a.bodyHeight()-2)
	top, topW := 0, float32(0)
	for i := lo; i < hi; i++ {
		w := row[i]
		if w > topW {
			top, topW = i, w
		}
		b.WriteString(fmt.Sprintf("  %-12s %s %5.1f%%\n", label(i), bar(float64(w), 24), w*100))
	}

	b.WriteString("\n" + dimStyle.Render(fmt.Sprintf(
		"  attends most to %s (%.0f%%)", fmt.Sprintf("%q", label(top)), topW*100)))
	if len(e.Weights) > 1 {
		var sink float64
		for i := 1; i < len(e.Weights); i++ {
			sink += float64(e.Weights[i][0])
		}
		sink /= float64(len(e.Weights) - 1)
		b.WriteString(dimStyle.Render(fmt.Sprintf(
			"\n  this head's sink: token 0 gets %.0f%% of later positions' attention on average", sink*100)))
	}
	return b.String()
}

// attentionTraceView follows one query token down through every layer at the
// selected head, showing which earlier token it leaned on hardest at each
// depth — whether that target holds steady or drifts as the representation
// deepens.
func (a *Inspect) attentionTraceView(s *inspectStep, q int, label func(int) string) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("  tracing %s   head %s\n\n",
		headingStyle.Render(truncate(fmt.Sprintf("%q", label(q)), 14)),
		keyStyle.Render(fmt.Sprint(a.head))))
	b.WriteString(headerRowStyle.Render(
		fmt.Sprintf("  %-5s %-14s %6s  %s", "layer", "attends to", "wt", "")) + "\n")

	nLayer := s.tr.Header.Config.NLayer
	lastTarget, firstStable := -1, -1
	for l := 0; l < nLayer; l++ {
		e, ok := s.tr.LayerEvent(l, trace.KindAttention, a.head)
		if !ok || q >= len(e.Weights) {
			continue
		}
		idx, w := argmax(e.Weights[q][:q+1])

		if idx == lastTarget {
			if firstStable < 0 {
				firstStable = l
			}
		} else {
			firstStable = -1
		}
		lastTarget = idx

		text := fmt.Sprintf("%q", label(idx))
		if idx == q {
			text += " (self)"
		}
		row := fmt.Sprintf("  %-5d %-14s %5.1f%%  %s", l, truncate(text, 14), w*100, bar(float64(w), 16))

		switch {
		case l == a.layer:
			b.WriteString(selectedStyle.Render(row))
		case l == firstStable:
			b.WriteString(headingStyle.Render(row))
		default:
			b.WriteString(row)
		}
		b.WriteString("\n")
	}

	if firstStable >= 0 {
		b.WriteString("\n" + dimStyle.Render(fmt.Sprintf(
			"  settles on %s from layer %d on (highlighted)",
			fmt.Sprintf("%q", label(lastTarget)), firstStable)))
	}
	return b.String()
}

// argmax is the index and value of the largest entry in v. Attention rows are
// never empty by the time this is called — q is clamped against a row that
// exists, and a trace row is always at least the self-attention entry.
func argmax(v []float32) (int, float32) {
	bi, bv := 0, v[0]
	for i, x := range v[1:] {
		if x > bv {
			bi, bv = i+1, x
		}
	}
	return bi, bv
}

// --- attribution --------------------------------------------------------------

// attributionView answers the question the attention grid can't: not what a head
// looked at, but whether it is why the answer came out the way it did. Each row
// is one component's push on the final token's logit, positive to the right.
//
// The two views are worth reading together — a head with a striking pattern and
// no push is doing something the output doesn't depend on, and that combination
// is common enough that the pattern alone is easy to over-read.
func (a *Inspect) attributionView() string {
	s := a.active()
	if s == nil {
		return ""
	}
	events := s.tr.Attributions(a.layer)
	if len(events) == 0 {
		return dimStyle.Render(
			"  No attribution in this trace.\n" +
				"  It's opt-in — trace.Opts{Attribution: true} — since it costs an extra\n" +
				"  projection per layer and an event per component.")
	}

	target, label := a.attributionTarget(s)
	if target < 0 {
		return dimStyle.Render("  Nothing to attribute against: this trace has no prediction.")
	}

	var b strings.Builder
	b.WriteString(fmt.Sprintf("  layer %s   pushing %s\n\n",
		keyStyle.Render(fmt.Sprint(a.layer)),
		headingStyle.Render(truncate(fmt.Sprintf("%q", label), 16))))

	// Bars are scaled within the layer, so a layer that barely moves anything
	// doesn't render as flat — the shape across its components is still the
	// thing worth seeing. The numbers carry the absolute size.
	var scale float64
	for _, e := range events {
		if v, ok := e.EffectOn(target); ok {
			scale = math.Max(scale, math.Abs(float64(v)))
		}
	}

	b.WriteString(headerRowStyle.Render(
		fmt.Sprintf("  %-10s %9s %9s  %s", "component", "Δlogit", "‖write‖", "")) + "\n")

	for _, e := range events {
		v, ok := e.EffectOn(target)
		if !ok {
			continue
		}
		row := fmt.Sprintf("  %-10s %+9.3f %9.3f  %s",
			componentName(e), v, e.Norm, signedBar(float64(v), scale, 12))
		if e.Component == trace.ComponentHead && e.Head == a.head {
			b.WriteString(selectedStyle.Render(row))
		} else {
			b.WriteString(row)
		}
		b.WriteString("\n")
	}

	b.WriteString("\n" + dimStyle.Render(
		"  every component's push, over every layer, sums to the output logit"))
	if top := a.topContributors(s, target, 4); len(top) > 0 {
		b.WriteString("\n" + dimStyle.Render("  largest across the whole pass: "+strings.Join(top, ", ")))
	}
	return b.String()
}

// attributionTarget is the token being attributed: whatever the pass finally
// predicted.
func (a *Inspect) attributionTarget(s *inspectStep) (int, string) {
	if final := a.finalPrediction(); final != nil {
		return final.ID, final.Text
	}
	for _, e := range s.lens {
		if e.TargetRank > 0 {
			return e.TargetID, e.TargetText
		}
	}
	return -1, ""
}

// topContributors ranks every component in the run by how hard it pushed the
// target, so a layer-by-layer view has somewhere to start.
func (a *Inspect) topContributors(s *inspectStep, target, n int) []string {
	type hit struct {
		label string
		v     float32
	}
	var hits []hit
	for _, e := range s.tr.Kind(trace.KindAttribution) {
		if v, ok := e.EffectOn(target); ok {
			hits = append(hits, hit{fmt.Sprintf("L%d %s", e.Layer, componentName(e)), v})
		}
	}
	sort.Slice(hits, func(i, j int) bool {
		return math.Abs(float64(hits[i].v)) > math.Abs(float64(hits[j].v))
	})
	if len(hits) > n {
		hits = hits[:n]
	}
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = fmt.Sprintf("%s %+.2f", h.label, h.v)
	}
	return out
}

func componentName(e trace.Event) string {
	switch e.Component {
	case trace.ComponentHead:
		return fmt.Sprintf("head %d", e.Head)
	case trace.ComponentEmbed:
		return "embedding"
	default:
		return string(e.Component)
	}
}

// signedBar draws a magnitude either side of a centre rule, so the sign is
// visible as direction before any number is read. Negative is drawn in the alert
// colour rather than on the magnitude ramp: a large push down and a large push
// up are equally strong readings, and putting both on brightness alone would
// make them indistinguishable at a glance.
func signedBar(v, scale float64, half int) string {
	if scale <= 0 {
		scale = 1
	}
	frac := math.Abs(v) / scale
	n := min(half, int(frac*float64(half)+0.5))
	rule := amber.NFg(amber.Rule).Render("│")
	gap := strings.Repeat(" ", half)

	if v < 0 {
		fill := lipgloss.NewStyle().Foreground(amber.Alert).
			Render(strings.Repeat("█", n))
		return strings.Repeat(" ", half-n) + fill + rule + gap
	}
	fill := lipgloss.NewStyle().Foreground(amber.Of(frac)).
		Render(strings.Repeat("█", n))
	return gap + rule + fill + strings.Repeat(" ", half-n)
}

// --- ablation -----------------------------------------------------------------

// ablationView compares the baseline run against a shadow run with the
// selected head's output forced to zero, layer by layer. It answers the
// question attribution can only guess at: does this head actually change the
// answer, or just look busy?
func (a *Inspect) ablationView() string {
	if len(a.steps) == 0 {
		return ""
	}
	base := a.steps[a.cur]
	if a.cur >= len(a.ablateSteps) {
		return dimStyle.Render(
			"  No ablation run yet.\n" +
				"  Select a layer and head (arrows), then press 'a' to force it to\n" +
				"  zero and compare its logit lens against this baseline.")
	}
	abl := a.ablateSteps[a.cur]

	var b strings.Builder
	b.WriteString(fmt.Sprintf("  ablating layer %s head %s\n\n",
		keyStyle.Render(fmt.Sprint(a.layer)), keyStyle.Render(fmt.Sprint(a.head))))
	b.WriteString(headerRowStyle.Render(fmt.Sprintf("  %-5s %-14s %7s   %-14s %7s   %s",
		"layer", "baseline", "prob", "ablated", "prob", "Δ baseline's pick")) + "\n")

	firstDiverge := -1
	n := min(len(base.lens), len(abl.lens))
	lo, hi := window(n, a.layer, a.bodyHeight()-1)
	for i := lo; i < hi; i++ {
		be, ae := base.lens[i], abl.lens[i]
		if len(be.Top) == 0 || len(ae.Top) == 0 {
			continue
		}
		bt, at := be.Top[0], ae.Top[0]
		if firstDiverge < 0 && bt.ID != at.ID {
			firstDiverge = be.Layer
		}

		// How much probability the ablated run still gives baseline's own top
		// pick — zero if it fell out of the ablated run's top-k entirely,
		// which is itself a meaningful answer (the head mattered a lot).
		var ablatedBaseProb float64
		for _, c := range ae.Top {
			if c.ID == bt.ID {
				ablatedBaseProb = c.Prob
				break
			}
		}
		delta := ablatedBaseProb - bt.Prob

		label := fmt.Sprintf("%d", be.Layer)
		if be.Layer >= base.tr.Header.Config.NLayer {
			label = "out"
		}

		row := fmt.Sprintf("  %-5s %-14s %6.1f%%   %-14s %6.1f%%   %s",
			label, truncate(fmt.Sprintf("%q", bt.Text), 14), bt.Prob*100,
			truncate(fmt.Sprintf("%q", at.Text), 14), at.Prob*100,
			signedBar(delta, 1, 10))

		switch {
		case be.Layer == a.layer:
			b.WriteString(selectedStyle.Render(row))
		case be.Layer == firstDiverge:
			b.WriteString(headingStyle.Render(row))
		default:
			b.WriteString(row)
		}
		b.WriteString("\n")
	}

	switch {
	case firstDiverge < 0:
		b.WriteString("\n" + dimStyle.Render(
			"  top prediction never changes — this head doesn't look causally load-bearing here"))
	case firstDiverge >= base.tr.Header.Config.NLayer:
		b.WriteString("\n" + dimStyle.Render(
			"  only the final output changes — every intermediate layer still agrees"))
	default:
		b.WriteString("\n" + dimStyle.Render(fmt.Sprintf(
			"  the top prediction first changes at layer %d (highlighted)", firstDiverge)))
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
