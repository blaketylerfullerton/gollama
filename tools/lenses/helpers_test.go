// Package lenses tests the mechinterp-facing surface of engine/model — head
// ablation and direct logit attribution — as an outside consumer, the same
// relationship tools/trace and tools/tui already have with the engine.
//
// The mechanism itself can't live here: ablation has to gate the actual copy
// into the residual stream inside Attention.Forward, and attribution needs
// direct access to the LM head's and final norm's weights as the pass runs.
// Both are physically wired into engine/model's hot path (see attention.go's
// pass.ablated and gpt.go's attribute). What belongs here is everything that
// only exercises GPT's exported API — which is all these tests ever did, even
// when they lived beside the engine's own tests.
package lenses

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
)

// tinyConfig mirrors engine/model's own test helper of the same name rather
// than importing it — it's unexported there, and small enough that
// duplicating it is cheaper than exporting test-only surface from the engine.
// The same tradeoff tools/tui's HeadRef already makes against model.HeadRef.
func tinyConfig() model.GPTConfig {
	return model.GPTConfig{
		VocabSize:    16,
		NLayer:       2,
		NHead:        4,
		NKVHead:      2,
		NEmbed:       8,
		HeadDim:      4,
		Intermediate: 16,
		RopeBase:     1e6,
		NormEps:      1e-6,
		TieEmbed:     true,
		SequenceLen:  32,
	}
}

// checkpointDir is where the Qwen3-0.6B download lands — two directories up
// from tools/lenses, same as it is two up from engine/model.
const checkpointDir = "../../checkpoints/qwen3-0.6b"

// realGPT loads the real checkpoint for the one test that checks attribution
// holds on trained weights rather than random ones. Skips rather than fails
// when the checkpoint isn't there, so a fresh clone still passes.
func realGPT(t *testing.T) *model.GPT {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping checkpoint test in -short mode")
	}
	if _, err := os.Stat(filepath.Join(checkpointDir, "model.safetensors")); err != nil {
		t.Skipf("no checkpoint at %s — run: huggingface-cli download Qwen/Qwen3-0.6B --local-dir %s",
			checkpointDir, checkpointDir)
	}
	gpt, err := model.FromDirectory(checkpointDir)
	if err != nil {
		t.Fatalf("loading the real checkpoint: %v", err)
	}
	return gpt
}

// tokenizerFor loads the real tokenizer beside the real checkpoint, for the
// one test that needs to encode an actual prompt.
func tokenizerFor(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.FromDirectory(checkpointDir)
	if err != nil {
		t.Fatalf("loading the tokenizer: %v", err)
	}
	return tok
}
