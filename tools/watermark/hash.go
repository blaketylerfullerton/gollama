// Package watermark implements a SynthID-Text-style watermark: a keyed
// tournament-sampling scheme that steers generation toward tokens a secret
// function scores highly, and a detector that recovers that bias from text
// alone. It has no dependency beyond engine/model — nothing here changes how
// GPT.Generate or model.Sampler work, this is a second, independent way to
// draw tokens from the same model.
package watermark

// mix64 is SplitMix64's finalizer — three xor-shifts and two multiplies that
// turn a counter-like input into something that looks uniformly random bit
// for bit. Every pseudorandom value below is built by feeding this a
// different combination of key, context, layer and token id: the mixing
// itself doesn't need to be cryptographic, just free of the kind of
// correlation that would make one token's score predictable from another's.
func mix64(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// contextSeed folds the watermark key and a window of preceding token ids
// into one number. Generate and Detect never share any other state — as
// long as both compute this the same way over the same context, they agree
// on every downstream g-value without coordinating anything else.
func contextSeed(key uint64, context []int) uint64 {
	h := key
	for _, id := range context {
		h = mix64(h ^ (uint64(uint32(id))*0x9e3779b97f4a7c15 + 1))
	}
	return h
}

// gValue is the pseudorandom score one tournament round assigns to one
// candidate token: deterministic given (seed, layer, token), uniform on
// [0,1). Salting with layer before hashing is what makes the L layers behave
// independently despite sharing a seed — round 2 isn't just round 1 again.
func gValue(seed uint64, layer, tokenID int) float64 {
	h := mix64(seed ^ mix64(uint64(layer)*0xff51afd7ed558ccd+uint64(uint32(tokenID))))
	return float64(h>>11) / float64(uint64(1)<<53)
}
