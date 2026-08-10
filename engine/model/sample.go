package model

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Candidate is one token and its probability after temperature scaling.
type Candidate struct {
	ID   int
	Prob float64
}

// SampleOpts controls how a token is chosen from a row of logits.
//
// The filters compose, and they are applied in this order: temperature scales
// the logits, softmax turns them into probabilities, then top-k and top-p each
// discard the tail. Whatever survives is renormalized and drawn from.
type SampleOpts struct {
	// Temperature divides the logits before softmax. Below ~1 sharpens the
	// distribution, above ~1 flattens it. Zero or negative means greedy: take
	// the argmax and ignore every other setting here.
	Temperature float64
	// TopK keeps only the k most likely tokens. Zero disables it.
	TopK int
	// TopP keeps the smallest set of tokens whose probabilities sum to at
	// least p (nucleus sampling). Zero or >= 1 disables it.
	TopP float64
	// Seed fixes the RNG so a run reproduces exactly.
	Seed uint64
}

// Sampler draws tokens from logit rows. It owns its RNG, so two samplers with
// the same seed produce the same sequence regardless of what else is running.
type Sampler struct {
	Opts SampleOpts
	rng  *rand.Rand
}

func NewSampler(opts SampleOpts) *Sampler {
	// PCG wants two seed words; deriving the second from the first keeps the
	// public API a single number.
	return &Sampler{
		Opts: opts,
		rng:  rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15)),
	}
}

// Sample picks one token id from a row of logits.
func (s *Sampler) Sample(logits []float32) int {
	if s.Opts.Temperature <= 0 {
		return Argmax(logits)
	}

	cands := softmaxCandidates(logits, s.Opts.Temperature)
	cands = applyTopK(cands, s.Opts.TopK)
	cands = applyTopP(cands, s.Opts.TopP)

	// The filters above dropped probability mass, so draw against what's left
	// rather than renormalizing the slice in place.
	var total float64
	for _, c := range cands {
		total += c.Prob
	}
	r := s.rng.Float64() * total
	for _, c := range cands {
		r -= c.Prob
		if r <= 0 {
			return c.ID
		}
	}
	// Only reachable through floating-point drift at the very end of the walk.
	return cands[len(cands)-1].ID
}

// Argmax returns the index of the largest logit — greedy decoding.
func Argmax(logits []float32) int {
	best := 0
	for i, l := range logits {
		if l > logits[best] {
			best = i
		}
	}
	return best
}

// TopCandidates returns the k most likely tokens after temperature scaling.
// Useful for showing a distribution; Sample runs the same path internally.
func TopCandidates(logits []float32, temperature float64, k int) []Candidate {
	if temperature <= 0 {
		temperature = 1
	}
	return applyTopK(softmaxCandidates(logits, temperature), k)
}

// softmaxCandidates applies temperature, softmaxes over the whole vocabulary,
// and returns every token sorted most likely first.
func softmaxCandidates(logits []float32, temperature float64) []Candidate {
	maxLogit := float64(logits[Argmax(logits)])

	out := make([]Candidate, len(logits))
	var sum float64
	for i, l := range logits {
		// Subtracting the max before exp keeps this from overflowing; it
		// cancels out in the division below.
		p := math.Exp((float64(l) - maxLogit) / temperature)
		out[i] = Candidate{ID: i, Prob: p}
		sum += p
	}
	for i := range out {
		out[i].Prob /= sum
	}

	sort.Slice(out, func(a, b int) bool {
		if out[a].Prob != out[b].Prob {
			return out[a].Prob > out[b].Prob
		}
		return out[a].ID < out[b].ID // stable ordering for equal probabilities
	})
	return out
}

func applyTopK(cands []Candidate, k int) []Candidate {
	if k <= 0 || k >= len(cands) {
		return cands
	}
	return cands[:k]
}

// applyTopP keeps the smallest prefix whose cumulative probability reaches p.
// cands must already be sorted most likely first.
func applyTopP(cands []Candidate, p float64) []Candidate {
	if p <= 0 || p >= 1 {
		return cands
	}
	var cum float64
	for i, c := range cands {
		cum += c.Prob
		if cum >= p {
			return cands[:i+1] // inclusive: the token that crossed p is kept
		}
	}
	return cands
}
