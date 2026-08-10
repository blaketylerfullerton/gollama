package main

import (
	"testing"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/tools/tui"
)

// The picker sizes models with its own copy of the parameter formula, because
// package tui doesn't import model/ — it has to describe checkpoints that
// aren't on disk, where there is no GPTConfig to hand it.
//
// That duplication is the deal, and this is what makes it safe: main imports
// both, so it can run the two formulas over the same shapes and fail here if
// they ever disagree. Without it, a change to one silently makes every number
// on the picker's memory panel wrong by a constant factor, and nothing on
// screen would look off.
func TestPickerParamFormulaMatchesTheModel(t *testing.T) {
	for _, tc := range []struct {
		name string
		arch tui.Arch
	}{
		{"qwen3-0.6b", tui.Arch{
			NLayer: 28, NHead: 16, NKVHead: 8, HeadDim: 128,
			NEmbed: 1024, Intermediate: 3072, VocabSize: 151936, TieEmbed: true,
		}},
		{"qwen3-8b, untied", tui.Arch{
			NLayer: 36, NHead: 32, NKVHead: 8, HeadDim: 128,
			NEmbed: 4096, Intermediate: 12288, VocabSize: 151936, TieEmbed: false,
		}},
		{"tiny, head dim not nembed/nhead", tui.Arch{
			NLayer: 2, NHead: 4, NKVHead: 2, HeadDim: 16,
			NEmbed: 32, Intermediate: 96, VocabSize: 50257, TieEmbed: true,
		}},
	} {
		cfg := model.GPTConfig{
			NLayer: tc.arch.NLayer, NHead: tc.arch.NHead, NKVHead: tc.arch.NKVHead,
			HeadDim: tc.arch.HeadDim, NEmbed: tc.arch.NEmbed,
			Intermediate: tc.arch.Intermediate, VocabSize: tc.arch.VocabSize,
			TieEmbed: tc.arch.TieEmbed,
		}
		if got, want := tc.arch.Params(), int64(paramTotal(cfg)); got != want {
			t.Errorf("%s: tui counts %d parameters, model counts %d", tc.name, got, want)
		}
	}
}

// The kv cache figure on the picker is the one that decides whether a long
// context fits, and model/ owns the real definition.
func TestPickerKVFormulaMatchesTheCache(t *testing.T) {
	cfg := model.GPTConfig{
		NLayer: 28, NHead: 16, NKVHead: 8, HeadDim: 128,
		NEmbed: 1024, Intermediate: 3072, VocabSize: 151936, TieEmbed: true,
	}
	arch := tui.Arch{
		NLayer: cfg.NLayer, NHead: cfg.NHead, NKVHead: cfg.NKVHead,
		HeadDim: cfg.HeadDim, NEmbed: cfg.NEmbed,
		Intermediate: cfg.Intermediate, VocabSize: cfg.VocabSize, TieEmbed: true,
	}
	if got, want := arch.KVBytesPerToken(), int64(model.NewKVCache(cfg).BytesPerToken()); got != want {
		t.Errorf("tui says %d bytes per cached token, model says %d", got, want)
	}
}

// An empty directory is how the picker says "the built-in random model", and a
// directory with no weights in it is a fresh clone. Both have to land on the
// demo rather than on an error.
func TestSetupFallsBackToTheRandomModel(t *testing.T) {
	for _, dir := range []string{"", t.TempDir()} {
		s, err := setup(dir, "hello")
		if err != nil {
			t.Fatalf("setup(%q): %v", dir, err)
		}
		if s.real {
			t.Errorf("setup(%q) reported a real checkpoint", dir)
		}
		if s.dir != "" {
			t.Errorf("setup(%q) recorded dir %q, want empty for the random model", dir, s.dir)
		}
	}
}
