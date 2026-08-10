package model

import (
	"math"
	"testing"
)

// maxDiff is the largest absolute difference between two logit rows.
func maxDiff(a, b []float32) (worst float64, at int) {
	for i := range a {
		if d := math.Abs(float64(a[i]) - float64(b[i])); d > worst {
			worst, at = d, i
		}
	}
	return worst, at
}

// The whole reason the uncached path was kept: cached decoding has to produce
// the same logits as recomputing from scratch. A wrong rotary position offset —
// the classic bug here — still runs and still yields fluent text, so only a
// comparison catches it.
func TestCachedMatchesUncachedIncrementally(t *testing.T) {
	gpt := tinyGPT(t)
	ids := []int{3, 7, 1, 4, 6, 2, 9}

	cache := NewKVCache(gpt.Config)

	// Feed one token at a time, checking against a full uncached pass over the
	// same prefix at every step. Position drift compounds, so a late-position
	// bug shows up here even if position 0 and 1 happen to agree.
	for n := 1; n <= len(ids); n++ {
		got, err := gpt.ForwardCached(ids[n-1:n], cache)
		if err != nil {
			t.Fatal(err)
		}
		full := gpt.Forward(ids[:n])
		want := full[len(full)-1]

		if worst, at := maxDiff(got, want); worst > 1e-4 {
			t.Fatalf("after %d tokens: logits differ by %g at index %d (cached %v, uncached %v)",
				n, worst, at, got[at], want[at])
		}
		if cache.Len() != n {
			t.Fatalf("cache holds %d positions, want %d", cache.Len(), n)
		}
	}
}

// Prefilling the whole prompt at once must agree with feeding it token by
// token — the two paths differ in how many rows attention handles per call.
func TestPrefillMatchesIncremental(t *testing.T) {
	gpt := tinyGPT(t)
	ids := []int{5, 2, 8, 1, 3}

	bulk := NewKVCache(gpt.Config)
	atOnce, err := gpt.ForwardCached(ids, bulk)
	if err != nil {
		t.Fatal(err)
	}

	oneByOne := NewKVCache(gpt.Config)
	var stepwise []float32
	for _, id := range ids {
		stepwise, err = gpt.ForwardCached([]int{id}, oneByOne)
		if err != nil {
			t.Fatal(err)
		}
	}

	if worst, at := maxDiff(atOnce, stepwise); worst > 1e-4 {
		t.Errorf("prefill and incremental differ by %g at index %d", worst, at)
	}
	if bulk.Len() != oneByOne.Len() {
		t.Errorf("cache lengths diverged: %d vs %d", bulk.Len(), oneByOne.Len())
	}
}

// Mixed batch sizes are what a real server does: prefill a prompt, decode a
// few, then accept more input.
func TestCachedHandlesMixedBatchSizes(t *testing.T) {
	gpt := tinyGPT(t)
	ids := []int{1, 2, 3, 4, 5, 6}

	cache := NewKVCache(gpt.Config)
	for _, chunk := range [][]int{ids[0:3], ids[3:4], ids[4:6]} {
		if _, err := gpt.ForwardCached(chunk, cache); err != nil {
			t.Fatal(err)
		}
	}

	got, err := gpt.ForwardCached([]int{7}, cache)
	if err != nil {
		t.Fatal(err)
	}
	full := gpt.Forward(append(ids, 7))
	if worst, at := maxDiff(got, full[len(full)-1]); worst > 1e-4 {
		t.Errorf("chunked feeding differs by %g at index %d", worst, at)
	}
}

func TestGenerateCachedMatchesUncached(t *testing.T) {
	gpt := tinyGPT(t)
	prompt := []int{2, 5, 1}
	opts := GenerateOpts{MaxTokens: 8, SampleOpts: SampleOpts{Temperature: 1, Seed: 9}}

	cached, err := gpt.Generate(prompt, opts)
	if err != nil {
		t.Fatal(err)
	}
	slow := opts
	slow.NoCache = true
	uncached, err := gpt.Generate(prompt, slow)
	if err != nil {
		t.Fatal(err)
	}

	if len(cached) != len(uncached) {
		t.Fatalf("cached produced %d tokens, uncached %d", len(cached), len(uncached))
	}
	for i := range cached {
		if cached[i] != uncached[i] {
			t.Fatalf("token %d: cached %d, uncached %d — the paths diverged",
				i, cached[i], uncached[i])
		}
	}
}

// Greedy decoding removes sampling from the picture, so any divergence is
// unambiguously the cache's fault.
func TestGenerateCachedMatchesUncachedGreedy(t *testing.T) {
	gpt := tinyGPT(t)
	prompt := []int{4, 4, 2}

	cached, err := gpt.Generate(prompt, GenerateOpts{MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	uncached, err := gpt.Generate(prompt, GenerateOpts{MaxTokens: 10, NoCache: true})
	if err != nil {
		t.Fatal(err)
	}
	for i := range cached {
		if i >= len(uncached) || cached[i] != uncached[i] {
			t.Fatalf("greedy diverged at token %d: %v vs %v", i, cached, uncached)
		}
	}
}

// --- cache bookkeeping ------------------------------------------------------

func TestCacheAccounting(t *testing.T) {
	cfg := tinyConfig()
	cache := NewKVCache(cfg)

	if cache.Len() != 0 || cache.Bytes() != 0 {
		t.Errorf("a fresh cache should be empty, got Len %d Bytes %d", cache.Len(), cache.Bytes())
	}

	// NLayer(2) x NKVHead(2) x HeadDim(4) x 2 tensors x 4 bytes = 128
	if got, want := cache.BytesPerToken(), 2*2*4*2*4; got != want {
		t.Errorf("BytesPerToken is %d, want %d", got, want)
	}

	gpt := tinyGPT(t)
	if _, err := gpt.ForwardCached([]int{1, 2, 3}, cache); err != nil {
		t.Fatal(err)
	}
	if cache.Len() != 3 {
		t.Errorf("Len is %d, want 3", cache.Len())
	}
	if got, want := cache.Bytes(), 3*cache.BytesPerToken(); got != want {
		t.Errorf("Bytes is %d, want %d", got, want)
	}
}

func TestCacheResetReusesStorage(t *testing.T) {
	gpt := tinyGPT(t)
	cache := NewKVCache(gpt.Config)

	if _, err := gpt.ForwardCached([]int{1, 2, 3, 4}, cache); err != nil {
		t.Fatal(err)
	}
	capBefore := cap(cache.layers[0].K[0])

	cache.Reset()
	if cache.Len() != 0 {
		t.Errorf("Len after Reset is %d, want 0", cache.Len())
	}
	if len(cache.layers[0].K[0]) != 0 {
		t.Error("Reset should truncate the stored keys")
	}
	if got := cap(cache.layers[0].K[0]); got != capBefore {
		t.Errorf("Reset reallocated: capacity went from %d to %d", capBefore, got)
	}

	// A reset cache must behave like a new one, not like a poisoned one.
	got, err := gpt.ForwardCached([]int{1, 2}, cache)
	if err != nil {
		t.Fatal(err)
	}
	full := gpt.Forward([]int{1, 2})
	if worst, at := maxDiff(got, full[len(full)-1]); worst > 1e-4 {
		t.Errorf("after Reset, logits differ by %g at index %d", worst, at)
	}
}

func TestForwardCachedRejectsBadInput(t *testing.T) {
	gpt := tinyGPT(t)

	if _, err := gpt.ForwardCached([]int{1}, nil); err == nil {
		t.Error("expected an error for a nil cache")
	}
	if _, err := gpt.ForwardCached(nil, NewKVCache(gpt.Config)); err == nil {
		t.Error("expected an error for no tokens")
	}
	if _, err := gpt.ForwardCached([]int{99999}, NewKVCache(gpt.Config)); err == nil {
		t.Error("expected an error for an out-of-vocab id")
	}

	// A cache built for a differently shaped model must be refused, not
	// silently indexed into.
	wrong := tinyConfig()
	wrong.NLayer = 1
	if _, err := gpt.ForwardCached([]int{1}, NewKVCache(wrong)); err == nil {
		t.Error("expected an error for a cache with the wrong layer count")
	}

	wrong = tinyConfig()
	wrong.NKVHead = 1
	if _, err := gpt.ForwardCached([]int{1}, NewKVCache(wrong)); err == nil {
		t.Error("expected an error for a cache with the wrong kv head count")
	}
}

// A caller-supplied cache should be reusable across calls, which is how a
// multi-turn conversation avoids reprocessing its own history.
func TestGenerateContinuesFromSuppliedCache(t *testing.T) {
	gpt := tinyGPT(t)
	cache := NewKVCache(gpt.Config)

	first, err := gpt.Generate([]int{1, 2}, GenerateOpts{MaxTokens: 3, Cache: cache})
	if err != nil {
		t.Fatal(err)
	}
	// 2 prompt tokens plus 3 generated, and the last generated token is never
	// fed back in — generation stops before that pass.
	if want := 2 + len(first) - 1; cache.Len() != want {
		t.Errorf("cache holds %d positions, want %d", cache.Len(), want)
	}
}

func TestCausalAttentionRejectsShortCache(t *testing.T) {
	// One query at position 5 needs 6 keys; giving it 2 is a bug worth a clear
	// panic rather than an index-out-of-range several frames deep.
	defer func() {
		if recover() == nil {
			t.Error("expected a panic when the cache is shorter than the query position")
		}
	}()
	q := [][]float32{{1, 0, 0, 0}}
	k := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	CausalAttention(q, k, k, 5)
}
