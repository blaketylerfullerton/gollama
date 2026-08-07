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
}

// Generate autoregressively continues the prompt and returns the new tokens.
// Stop tokens end the run and are not included in the result.
//
// This recomputes the entire prefix on every step, which is O(T^2) work over
// the run. That is deliberately the simple version: it is the reference a KV
// cache has to reproduce exactly, and the baseline any speedup is measured
// against. Don't optimize it here — add the cache as a separate path.
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

	stop := make(map[int]bool, len(opts.Stop)+len(g.Config.EOSTokenIDs))
	for _, id := range opts.Stop {
		stop[id] = true
	}
	for _, id := range g.Config.EOSTokenIDs {
		stop[id] = true
	}

	sampler := NewSampler(opts.SampleOpts)

	// Copy so appending never writes into the caller's backing array.
	seq := make([]int, len(prompt), len(prompt)+opts.MaxTokens)
	copy(seq, prompt)

	out := make([]int, 0, opts.MaxTokens)
	for len(out) < opts.MaxTokens {
		if g.Config.SequenceLen > 0 && len(seq) >= g.Config.SequenceLen {
			break // out of context
		}

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
