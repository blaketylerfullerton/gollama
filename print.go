package main

import (
	"fmt"
	"math"
	"strings"
)

// previewDims is how many leading dimensions of a vector we show. Anything
// past that is elided — we only need enough columns to eyeball the change.
const previewDims = 8

// maxGridTokens caps the attention grid so a long prompt doesn't flood the
// terminal with a matrix nobody can read.
const maxGridTokens = 10

// colHeader prints the dimension indices so they line up over vecRow's columns.
func colHeader(n int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "  %-7s", "dim")
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "%9d", i)
	}
	return b.String()
}

// vecRow formats the first n values of v as fixed-width signed columns.
func vecRow(label string, v []float32, n int) string {
	if n > len(v) {
		n = len(v)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "  %-7s", label)
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, "%+9.4f", v[i])
	}
	if len(v) > n {
		fmt.Fprintf(&b, "   … %d dims", len(v))
	}
	return b.String()
}

// PrintVecDiff shows before/after/delta for one vector plus a few summary
// stats. For rotary the norm should be unchanged — that's the sanity check.
func PrintVecDiff(title string, before, after []float32) {
	delta := make([]float32, len(before))
	maxIdx := 0
	for i := range before {
		delta[i] = after[i] - before[i]
		if math.Abs(float64(delta[i])) > math.Abs(float64(delta[maxIdx])) {
			maxIdx = i
		}
	}

	fmt.Println(title)
	fmt.Println(colHeader(previewDims))
	fmt.Println(vecRow("before", before, previewDims))
	fmt.Println(vecRow("after", after, previewDims))
	fmt.Println(vecRow("Δ", delta, previewDims))
	fmt.Printf("  ‖v‖ %.4f → %.4f   cos(before,after) %+.4f   max|Δ| %.4f @ dim %d\n",
		norm(before), norm(after), cosSim(before, after), math.Abs(float64(delta[maxIdx])), maxIdx)
	fmt.Println("  rotation preserves length — only the direction turns, by an angle set by position")
}

// PrintRotaryTable dumps the cos/sin lookup tables one row per position.
// Row 0 should be all cos=1 / sin=0 (no rotation), and columns to the right
// rotate more slowly than columns on the left.
func PrintRotaryTable(cos, sin [][]float32, maxPos int) {
	if maxPos > len(cos) {
		maxPos = len(cos)
	}
	fmt.Printf("rotary tables: %d positions x %d freq pairs\n", len(cos), len(cos[0]))
	fmt.Println(colHeader(previewDims))
	for t := 0; t < maxPos; t++ {
		fmt.Println(vecRow(fmt.Sprintf("cos t=%d", t), cos[t], previewDims))
		fmt.Println(vecRow(fmt.Sprintf("sin t=%d", t), sin[t], previewDims))
	}
	if maxPos < len(cos) {
		fmt.Printf("  … %d more positions\n", len(cos)-maxPos)
	}
	fmt.Println("  low dims turn fast (fine position detail), high dims turn slowly (coarse)")
}

func norm(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return math.Sqrt(s)
}

func cosSim(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	d := norm(a) * norm(b)
	if d == 0 {
		return 0
	}
	return dot / d
}

// meanNorm is the average vector length across positions — a single number for
// "how big is the residual stream right now".
func meanNorm(x [][]float32) float64 {
	if len(x) == 0 {
		return 0
	}
	var sum float64
	for _, row := range x {
		sum += norm(row)
	}
	return sum / float64(len(x))
}

// --- the walkthrough tracer -------------------------------------------------

// walkthrough implements model.Tracer by narrating a forward pass to stdout.
//
// A full trace of every layer would be unreadable, so it prints detail for one
// block and one head, and records a magnitude for every stage to summarize at
// the end.
type walkthrough struct {
	labels      []string // token text per position, for grid axes
	detailLayer int
	detailHead  int
	stages      []stageStat
}

type stageStat struct {
	layer int
	name  string
	norm  float64
}

// verbose reports whether this layer gets the full treatment. Layer -1 is used
// for stages outside the block stack, which always print.
func (w *walkthrough) verbose(layer int) bool {
	return layer < 0 || layer == w.detailLayer
}

func (w *walkthrough) Stage(layer int, name string, x [][]float32) {
	w.stages = append(w.stages, stageStat{layer, name, meanNorm(x)})

	if !w.verbose(layer) {
		return
	}
	label := name
	if layer >= 0 {
		label = fmt.Sprintf("layer %d: %s", layer, name)
	}
	fmt.Printf("\n%s — (%d tokens x %d dims), mean ‖x‖ %.4f\n",
		label, len(x), len(x[0]), meanNorm(x))
	fmt.Println(colHeader(previewDims))
	fmt.Println(vecRow("tok 0", x[0], previewDims))
}

func (w *walkthrough) Note(layer int, format string, args ...any) {
	if w.verbose(layer) {
		fmt.Printf("  %s\n", fmt.Sprintf(format, args...))
	}
}

func (w *walkthrough) Rotary(layer, head int, before, after []float32) {
	if layer == w.detailLayer && head == w.detailHead {
		fmt.Println()
		PrintVecDiff(fmt.Sprintf("rotary on q — layer %d, head %d, last position", layer, head),
			before, after)
	}
}

func (w *walkthrough) Attention(layer, head int, weights [][]float32) {
	if layer != w.detailLayer || head != w.detailHead {
		return
	}

	T := len(weights)
	n := T
	if n > maxGridTokens {
		n = maxGridTokens
	}

	fmt.Printf("\nattention weights — layer %d, head %d (each row attends across the columns)\n",
		layer, head)

	var hdr strings.Builder
	fmt.Fprintf(&hdr, "  %-9s", "")
	for j := 0; j < n; j++ {
		fmt.Fprintf(&hdr, "%9s", w.label(j))
	}
	fmt.Println(hdr.String())

	for i := 0; i < n; i++ {
		var row strings.Builder
		fmt.Fprintf(&row, "  %-9s", w.label(i))
		for j := 0; j < n; j++ {
			if j > i {
				// Not "zeroed" — these scores are never computed at all.
				fmt.Fprintf(&row, "%9s", "·")
			} else {
				fmt.Fprintf(&row, "%9.3f", weights[i][j])
			}
		}
		fmt.Println(row.String())
	}
	if T > n {
		fmt.Printf("  … %d more positions\n", T-n)
	}
	fmt.Println("  · = masked. Each row sums to 1, and token 0 can only ever attend to itself")
}

// label renders a token as a short, terminal-safe column heading.
func (w *walkthrough) label(i int) string {
	if i >= len(w.labels) {
		return fmt.Sprintf("t%d", i)
	}
	s := w.labels[i]
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\t", "\\t")
	s = strings.ReplaceAll(s, " ", "_")
	r := []rune(s)
	if len(r) > 8 {
		return string(r[:8])
	}
	return string(r)
}

// PrintSummary shows every stage's magnitude in order. Reading it top to
// bottom is the clearest picture of what a transformer actually does to a
// vector: branch off, normalize, add the result back, repeat.
func (w *walkthrough) PrintSummary() {
	var prev float64
	for _, s := range w.stages {
		label := s.name
		if s.layer >= 0 {
			label = fmt.Sprintf("layer %d: %s", s.layer, s.name)
		}
		if !isResidual(s.name) {
			// Indent the branches so the residual stream reads as one column.
			fmt.Printf("      %-30s %12.4f\n", label, s.norm)
			continue
		}
		delta := ""
		if prev != 0 {
			delta = fmt.Sprintf("   %+.4f", s.norm-prev)
		}
		fmt.Printf("  ▸  %-30s %12.4f%s\n", label, s.norm, delta)
		prev = s.norm
	}
	fmt.Println("\n  ▸ marks the residual stream itself — follow that column down and it only")
	fmt.Println("    grows, because pre-norm never rescales the stream, it just adds to it.")
	fmt.Println("    Indented rows are branches off the stream: a norm feeds each sublayer")
	fmt.Println("    (hence ‖x‖ ≈ √dims there), and the logits are the final projection out.")
}

// isResidual distinguishes the residual stream from the intermediates that
// branch off it. Matching on the stage name keeps this a display concern —
// the model doesn't need to know how we choose to lay the table out.
func isResidual(name string) bool {
	return name == "token embeddings" || strings.HasSuffix(name, "residual")
}
