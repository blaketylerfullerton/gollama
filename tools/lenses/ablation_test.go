package lenses

import (
	"testing"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
)

// A nil ablate set is what every existing caller passes (via ForwardCached),
// so it must be a true no-op — this is the regression guard for turning
// ForwardCached into a thin wrapper around ForwardCachedAblated.
func TestForwardCachedAblatedNilMatchesForwardCached(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{1, 5, 9}

	want, err := gpt.ForwardCached(ids, model.NewKVCache(cfg))
	if err != nil {
		t.Fatal(err)
	}
	got, err := gpt.ForwardCachedAblated(ids, model.NewKVCache(cfg), nil)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(want) {
		t.Fatalf("got %d logits, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit %d: got %v, want %v (a nil ablate set must not change the result)",
				i, got[i], want[i])
		}
	}
}

// Silencing every attention head in every layer must actually move the
// output — otherwise the ablation hook isn't wired into the forward pass at
// all, just accepted and ignored.
func TestAblatingEveryHeadChangesLogits(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{1, 5, 9, 2}

	base, err := gpt.ForwardCached(ids, model.NewKVCache(cfg))
	if err != nil {
		t.Fatal(err)
	}

	var all []model.HeadRef
	for l := 0; l < cfg.NLayer; l++ {
		for h := 0; h < cfg.NHead; h++ {
			all = append(all, model.HeadRef{Layer: l, Head: h})
		}
	}
	ablated, err := gpt.ForwardCachedAblated(ids, model.NewKVCache(cfg), all)
	if err != nil {
		t.Fatal(err)
	}

	for i := range base {
		if base[i] != ablated[i] {
			return // found a difference, as expected
		}
	}
	t.Fatal("ablating every attention head in every layer left the logits unchanged")
}

// Ablation must be scoped to exactly the requested head: the silenced head's
// own recorded write should vanish, but its layer-mates must not.
func TestAblationIsScopedToItsOwnHead(t *testing.T) {
	cfg := tinyConfig()
	gpt, err := model.NewRandomGPT(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{topK: 2}
	gpt.Trace = rec
	defer func() { gpt.Trace = nil }()

	ids := []int{1, 5, 9}
	if _, err := gpt.ForwardCachedAblated(ids, model.NewKVCache(cfg), []model.HeadRef{{Layer: 0, Head: 0}}); err != nil {
		t.Fatal(err)
	}

	var sawOtherHeadEffect bool
	for _, g := range rec.got {
		if g.layer != 0 || g.component < 0 {
			continue // not an attention head in the ablated layer
		}
		if g.component == 0 {
			if g.norm != 0 {
				t.Errorf("ablated head reported write norm %v, want 0", g.norm)
			}
			for j, e := range g.effects {
				if e != 0 {
					t.Errorf("ablated head reported nonzero effect on token %d: %v", rec.tokens[j], e)
				}
			}
			continue
		}
		for _, e := range g.effects {
			if e != 0 {
				sawOtherHeadEffect = true
			}
		}
	}
	if !sawOtherHeadEffect {
		t.Error("every other head in layer 0 reported zero effect too — " +
			"ablation may be silencing more than the one requested head")
	}
}
