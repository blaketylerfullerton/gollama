package main

import (
	"fmt"
	"math"
	"strings"
)

// previewDims is how many leading dimensions of a vector we show. Anything
// past that is elided — we only need enough columns to eyeball the change.
const previewDims = 8

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
	fmt.Printf("  ‖v‖ %.4f → %.4f   cos(before,after) %+.4f   max|Δ| %.4f @ dim %d\n\n",
		norm(before), norm(after), cosSim(before, after), math.Abs(float64(delta[maxIdx])), maxIdx)
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
	fmt.Println()
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
