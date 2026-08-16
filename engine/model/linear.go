package model

import (
	"runtime"
	"sync"
)

// NewRandomLinear creates a randomly initialized Linear layer with no bias.
// Weight is (out, in) row-major. Qwen3's attention and MLP projections are all
// bias-free, so Bias stays nil — MatMul skips it.
func NewRandomLinear(in, out int) Linear {
	return Linear{
		Weight: randFloats(in * out),
		Bias:   nil,
		In:     in,
		Out:    out,
	}
}

// matMulParallelThreshold is the total multiply-add count (T*out*in) below
// which spinning up goroutines costs more than it saves.
const matMulParallelThreshold = 1 << 16

// MatMul projects x (T, in) through a Linear layer's weights, computing
// x @ W^T. Returns (T, out).
//
// Every (t, o) output element is independent. During decode T is 1 — a
// single token — so splitting work across rows wouldn't parallelize anything;
// instead the output columns are split across GOMAXPROCS goroutines. That's
// also the split that matters most: the LM head projection (out == VocabSize,
// on the order of 100k+) dominates per-token decode cost.
func MatMul(x [][]float32, lin Linear) [][]float32 {
	out := make([][]float32, len(x))
	for t := range x {
		out[t] = make([]float32, lin.Out)
	}

	if len(x)*lin.Out*lin.In < matMulParallelThreshold {
		for t, row := range x {
			matMulRange(row, lin, out[t], 0, lin.Out)
		}
		return out
	}

	workers := min(runtime.GOMAXPROCS(0), lin.Out)
	if workers < 1 {
		workers = 1
	}

	var wg sync.WaitGroup
	chunk := (lin.Out + workers - 1) / workers
	for w := 0; w < workers; w++ {
		oStart := w * chunk
		oEnd := oStart + chunk
		if oStart >= lin.Out {
			break
		}
		if oEnd > lin.Out {
			oEnd = lin.Out
		}
		wg.Add(1)
		go func(oStart, oEnd int) {
			defer wg.Done()
			for t, row := range x {
				matMulRange(row, lin, out[t], oStart, oEnd)
			}
		}(oStart, oEnd)
	}
	wg.Wait()
	return out
}

// matMulRange computes dst[o] = row . weightRow(o) [+ bias[o]] for o in [oStart, oEnd).
func matMulRange(row []float32, lin Linear, dst []float32, oStart, oEnd int) {
	for o := oStart; o < oEnd; o++ {
		weightRow := lin.Weight[o*lin.In : (o+1)*lin.In]
		sum := dotProduct(row, weightRow)
		if lin.Bias != nil {
			sum += lin.Bias[o]
		}
		dst[o] = sum
	}
}
