package watermark

import (
	"fmt"
	"math/rand/v2"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
)

// Config is the shared secret and shape of a watermark. Generate and Detect
// only agree with each other when both are given the same Config — anyone
// missing the Key sees ordinary-looking text with no way to reconstruct the
// g-values that would reveal it.
type Config struct {
	// Key is the watermark's secret.
	Key uint64
	// ContextSize is how many preceding tokens seed each step's tournament —
	// h in the SynthID-Text paper. A step with fewer than ContextSize tokens
	// behind it (near the very start of a sequence) seeds from whatever
	// history it has instead of waiting for a full window.
	ContextSize int
	// Layers is the tournament depth — L in the paper. Each generated token
	// is the survivor of Layers single-elimination rounds among 2^Layers
	// i.i.d. draws from the model's own distribution, so raising it costs
	// more draws per token in exchange for a stronger, more confidently
	// detected signal.
	Layers int
}

// candidates is how many i.i.d. draws enter the tournament each step —
// 2^Layers, one bracket slot per draw.
func (c Config) candidates() int { return 1 << uint(c.Layers) }

// GenerateOpts is model.GenerateOpts's counterpart for tournament sampling.
// There is no TopK/TopP here: either would truncate the pool the tournament
// draws from before the watermark gets a say, which is exactly the
// distortion tournament sampling is meant to avoid.
type GenerateOpts struct {
	// Temperature shapes the underlying draw the same way it does for
	// ordinary sampling. Zero is treated as 1 (no scaling) rather than
	// greedy — greedy decoding has no randomness for a tournament to bias.
	Temperature float64
	MaxTokens   int
	Stop        []int
	OnToken     func(id int)
	// Seed drives which candidates get drawn into each tournament. It's
	// independent of Config.Key: two runs with the same Seed but different
	// Key draw the same candidate pools but the watermark picks different
	// winners from them.
	Seed uint64
}

// Generate autoregressively continues prompt using tournament sampling: each
// step draws cfg.candidates() i.i.d. samples from gpt's next-token
// distribution, then runs a single-elimination bracket where round l is
// decided by comparing gValue(seed, l, ·) between the two competitors. The
// survivor is more likely to have high g-values than a token drawn plainly
// would be — that bias is what Detect looks for.
//
// It drives gpt.ForwardCached itself rather than calling gpt.Generate,
// since Generate is hardwired to model.Sampler and offers no way to
// intervene in which candidate wins.
func Generate(gpt *model.GPT, cfg Config, prompt []int, opts GenerateOpts) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("watermark: prompt is empty, there is nothing to continue")
	}
	if opts.MaxTokens < 0 {
		return nil, fmt.Errorf("watermark: MaxTokens is %d, must not be negative", opts.MaxTokens)
	}
	temp := opts.Temperature
	if temp <= 0 {
		temp = 1
	}

	stop := make(map[int]bool, len(opts.Stop)+len(gpt.Config.EOSTokenIDs))
	for _, id := range opts.Stop {
		stop[id] = true
	}
	for _, id := range gpt.Config.EOSTokenIDs {
		stop[id] = true
	}

	rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15))
	cache := model.NewKVCache(gpt.Config)

	logits, err := gpt.ForwardCached(prompt, cache)
	if err != nil {
		return nil, err
	}

	seq := append([]int{}, prompt...)
	out := make([]int, 0, opts.MaxTokens)
	for len(out) < opts.MaxTokens {
		next := pickTournament(logits, temp, cfg, seq, rng)
		if stop[next] {
			break
		}
		out = append(out, next)
		seq = append(seq, next)
		if opts.OnToken != nil {
			opts.OnToken(next)
		}
		if len(out) == opts.MaxTokens {
			break // don't pay for a forward pass whose logits we'd discard
		}
		logits, err = gpt.ForwardCached([]int{next}, cache)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// pickTournament draws cfg.candidates() i.i.d. samples from softmax(logits/temp)
// and runs the bracket, returning the survivor.
func pickTournament(logits []float32, temp float64, cfg Config, seq []int, rng *rand.Rand) int {
	pool := model.TopCandidates(logits, temp, len(logits)) // full distribution, sorted

	survivors := make([]int, cfg.candidates())
	for i := range survivors {
		survivors[i] = weightedDraw(pool, rng)
	}

	ctx := seq
	if len(ctx) > cfg.ContextSize {
		ctx = ctx[len(ctx)-cfg.ContextSize:]
	}
	seed := contextSeed(cfg.Key, ctx)

	for layer := 0; layer < cfg.Layers; layer++ {
		next := make([]int, len(survivors)/2)
		for i := range next {
			a, b := survivors[2*i], survivors[2*i+1]
			if gValue(seed, layer, a) >= gValue(seed, layer, b) {
				next[i] = a
			} else {
				next[i] = b
			}
		}
		survivors = next
	}
	return survivors[0]
}

// weightedDraw samples one token id from pool, which TopCandidates already
// returns normalized and sorted most-likely-first.
func weightedDraw(pool []model.Candidate, rng *rand.Rand) int {
	r := rng.Float64()
	var cum float64
	for _, c := range pool {
		cum += c.Prob
		if r <= cum {
			return c.ID
		}
	}
	return pool[len(pool)-1].ID // floating-point drift at the very end of the walk
}
