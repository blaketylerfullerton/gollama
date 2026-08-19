package watermark

import "math"

// Score is what Detect reports about one token sequence.
type Score struct {
	// Positions is how many tokens in the sequence had a full ContextSize
	// window behind them and so could be scored at all.
	Positions int
	// MeanG is the mean g-value the emitted tokens actually received,
	// averaged over every tournament layer and every scored position.
	MeanG float64
	// Z is MeanG turned into a one-sided z-score against the null of
	// unwatermarked text, where each g-value behaves as an independent
	// Uniform[0,1) draw and MeanG should sit at 0.5.
	Z float64
}

// Detect recomputes, for every position in ids with enough preceding
// context, the g-value each tournament layer assigned to the token that was
// actually emitted there — the same values Generate's bracket competed on —
// and averages them.
//
// A token that survived a Generate tournament won its bracket by having
// higher-than-average g-values, so watermarked text's MeanG sits above 0.5
// and the gap widens (Z grows) with more tokens. Ordinary text has no reason
// to prefer high-g tokens, so its MeanG hovers at 0.5 with only sampling
// noise around it.
//
// ids should be the full sequence a token was generated into — prompt plus
// output — since each position's context comes from what actually preceded
// it, not from the output alone.
func Detect(cfg Config, ids []int) Score {
	var sum float64
	var positions int
	for i := cfg.ContextSize; i < len(ids); i++ {
		seed := contextSeed(cfg.Key, ids[i-cfg.ContextSize:i])
		for layer := 0; layer < cfg.Layers; layer++ {
			sum += gValue(seed, layer, ids[i])
		}
		positions++
	}
	if positions == 0 {
		return Score{}
	}
	terms := positions * cfg.Layers
	mean := sum / float64(terms)
	// Under the null, every term is one Uniform[0,1) draw — variance 1/12 —
	// and independent of every other term, so the mean of `terms` of them
	// has variance (1/12)/terms regardless of how they split between
	// positions and layers.
	stddev := math.Sqrt(1.0 / 12.0 / float64(terms))
	return Score{Positions: positions, MeanG: mean, Z: (mean - 0.5) / stddev}
}
