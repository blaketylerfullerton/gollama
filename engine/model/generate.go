package model

import "fmt"

// GenerateOpts controls a generation run.
type GenerateOpts struct {
	SampleOpts

	// MaxTokens caps how many tokens to produce.
	MaxTokens int
	// Stop lists token ids that end generation. The checkpoint's own
	// eos_token_id values are always honored in addition to these.
	Stop []int
	// OnToken, when set, is called with each token as it is produced — enough
	// to stream output instead of waiting for the whole completion.
	OnToken func(id int)
	// Cache lets a caller supply (and afterwards inspect) the KV cache, or
	// continue a previous run. When nil a fresh one is allocated per call.
	Cache *KVCache
	// NoCache forces the O(T²) uncached path, recomputing the whole prefix on
	// every step. Only useful for checking the cached path against it.
	NoCache bool
}

// Generate autoregressively continues the prompt and returns the new tokens.
// Stop tokens end the run and are not included in the result.
//
// The prompt is processed in a single prefill pass, then each new token is fed
// in on its own with the cache carrying the history forward.
func (g *GPT) Generate(prompt []int, opts GenerateOpts) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("generate: prompt is empty, there is nothing to continue")
	}
	if opts.MaxTokens < 0 {
		return nil, fmt.Errorf("generate: MaxTokens is %d, must not be negative", opts.MaxTokens)
	}
	for _, id := range prompt {
		if id < 0 || id >= g.Config.VocabSize {
			return nil, fmt.Errorf("generate: prompt contains token id %d, outside vocab of %d",
				id, g.Config.VocabSize)
		}
	}

	// The prompt plus everything generated has to fit the context window.
	budget := opts.MaxTokens
	if g.Config.SequenceLen > 0 {
		room := max(0, g.Config.SequenceLen-len(prompt))
		budget = min(budget, room)
	}

	stop := make(map[int]bool, len(opts.Stop)+len(g.Config.EOSTokenIDs))
	for _, id := range opts.Stop {
		stop[id] = true
	}
	for _, id := range g.Config.EOSTokenIDs {
		stop[id] = true
	}

	sampler := NewSampler(opts.SampleOpts)
	out := make([]int, 0, budget)

	if opts.NoCache {
		return g.generateUncached(prompt, budget, stop, sampler, opts, out)
	}

	cache := opts.Cache
	if cache == nil {
		cache = NewKVCache(g.Config)
	}

	// Prefill: the whole prompt in one pass, which fills the cache and yields
	// the distribution for the first new token.
	logits, err := g.ForwardCached(prompt, cache)
	if err != nil {
		return nil, err
	}

	for len(out) < budget {
		next := sampler.Sample(logits)
		if stop[next] {
			break
		}
		out = append(out, next)
		if opts.OnToken != nil {
			opts.OnToken(next)
		}
		if len(out) == budget {
			break // don't pay for a forward pass whose logits we'd discard
		}

		// Decode: one token in, and the cache supplies every key and value
		// behind it.
		logits, err = g.ForwardCached([]int{next}, cache)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// generateUncached recomputes the entire prefix on every step. It exists as the
// reference the cached path is verified against — nothing else should use it.
func (g *GPT) generateUncached(prompt []int, budget int, stop map[int]bool,
	sampler *Sampler, opts GenerateOpts, out []int) ([]int, error) {

	seq := make([]int, len(prompt), len(prompt)+budget)
	copy(seq, prompt)

	for len(out) < budget {
		logits := g.Forward(seq)
		next := sampler.Sample(logits[len(logits)-1])
		if stop[next] {
			break
		}
		seq = append(seq, next)
		out = append(out, next)
		if opts.OnToken != nil {
			opts.OnToken(next)
		}
	}
	return out, nil
}
