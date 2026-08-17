package model

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// tinyConfig is a Qwen3-shaped model small enough to write out by hand. The
// two properties that matter are inherited from the real thing:
//
//	HeadDim (4) != NEmbed/NHead (8/4 = 2)
//	NKVHead (2) < NHead (4)
//
// so anything that wrongly derives HeadDim, or that confuses q width with kv
// width, fails here rather than in a 600MB checkpoint.
func tinyConfig() GPTConfig {
	return GPTConfig{
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

// ramp produces deterministic, bounded, non-uniform values. Real weights would
// be random, but a fixed pattern keeps the test reproducible while still
// catching transposes — a symmetric or constant fill would not.
func ramp(n int, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i*7+seed*13)) * 0.05)
	}
	return out
}

// writeTinyCheckpoint emits a complete Qwen3 checkpoint using HF's real tensor
// names, plus the matching config.json.
func writeTinyCheckpoint(t *testing.T, dir string, cfg GPTConfig) {
	t.Helper()

	var tensors []stTensor
	seed := 0
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		seed++
		tensors = append(tensors, stTensor{name, "F32", shape, f32Bytes(ramp(n, seed)...)})
	}

	add("model.embed_tokens.weight", cfg.VocabSize, cfg.NEmbed)
	for i := 0; i < cfg.NLayer; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		add(p+"input_layernorm.weight", cfg.NEmbed)
		add(p+"self_attn.q_proj.weight", cfg.QOut(), cfg.NEmbed)
		add(p+"self_attn.k_proj.weight", cfg.KVOut(), cfg.NEmbed)
		add(p+"self_attn.v_proj.weight", cfg.KVOut(), cfg.NEmbed)
		add(p+"self_attn.o_proj.weight", cfg.NEmbed, cfg.QOut())
		add(p+"self_attn.q_norm.weight", cfg.HeadDim)
		add(p+"self_attn.k_norm.weight", cfg.HeadDim)
		add(p+"post_attention_layernorm.weight", cfg.NEmbed)
		add(p+"mlp.gate_proj.weight", cfg.Intermediate, cfg.NEmbed)
		add(p+"mlp.up_proj.weight", cfg.Intermediate, cfg.NEmbed)
		add(p+"mlp.down_proj.weight", cfg.NEmbed, cfg.Intermediate)
	}
	add("model.norm.weight", cfg.NEmbed)
	// No lm_head.weight: this checkpoint is tied, like Qwen3-0.6B.

	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), tensors)

	config := fmt.Sprintf(`{
		"vocab_size": %d, "hidden_size": %d, "intermediate_size": %d,
		"num_hidden_layers": %d, "num_attention_heads": %d,
		"num_key_value_heads": %d, "head_dim": %d,
		"rms_norm_eps": 1e-06, "rope_theta": 1000000,
		"tie_word_embeddings": true, "max_position_embeddings": %d}`,
		cfg.VocabSize, cfg.NEmbed, cfg.Intermediate, cfg.NLayer,
		cfg.NHead, cfg.NKVHead, cfg.HeadDim, cfg.SequenceLen)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
}

// HasWeights has to recognise a sharded checkpoint, not just a single
// model.safetensors — every real Qwen3 above 0.6B ships sharded, and a
// checker that only looked for the single-file name reported a fully and
// correctly downloaded checkpoint as though the directory were empty, which
// sent callers quietly down whatever "no checkpoint" path they had instead
// of loading the real weights right there on disk.
func TestHasWeights(t *testing.T) {
	t.Run("single file", func(t *testing.T) {
		dir := t.TempDir()
		writeSafetensors(t, filepath.Join(dir, "model.safetensors"), []stTensor{
			{"w", "F32", []int{1}, f32Bytes(1)},
		})
		if !HasWeights(dir) {
			t.Error("HasWeights = false for a directory with model.safetensors")
		}
	})

	t.Run("sharded", func(t *testing.T) {
		dir := t.TempDir()
		writeSafetensors(t, filepath.Join(dir, "model-00001-of-00001.safetensors"), []stTensor{
			{"w", "F32", []int{1}, f32Bytes(1)},
		})
		index := `{"weight_map":{"w":"model-00001-of-00001.safetensors"}}`
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
			t.Fatal(err)
		}
		if !HasWeights(dir) {
			t.Error("HasWeights = false for a sharded checkpoint with no single model.safetensors")
		}
	})

	t.Run("empty", func(t *testing.T) {
		if HasWeights(t.TempDir()) {
			t.Error("HasWeights = true for a directory with no weights at all")
		}
	})
}

func TestFromDirectory(t *testing.T) {
	dir := t.TempDir()
	cfg := tinyConfig()
	writeTinyCheckpoint(t, dir, cfg)

	gpt, err := FromDirectory(dir)
	if err != nil {
		t.Fatal(err)
	}

	if gpt.Config.HeadDim != 4 {
		t.Errorf("HeadDim: got %d, want 4 (must come from config, not NEmbed/NHead)", gpt.Config.HeadDim)
	}
	if len(gpt.Blocks) != cfg.NLayer {
		t.Fatalf("got %d blocks, want %d", len(gpt.Blocks), cfg.NLayer)
	}

	// Tied embeddings: the LM head must alias the embedding table, not a copy.
	if len(gpt.LMHead.Weight) != len(gpt.WTE) {
		t.Errorf("tied lm_head should reuse WTE: got %d weights, want %d",
			len(gpt.LMHead.Weight), len(gpt.WTE))
	}
	if &gpt.LMHead.Weight[0] != &gpt.WTE[0] {
		t.Error("tied lm_head should alias WTE rather than duplicate it")
	}

	attn := gpt.Blocks[0].Attn
	if attn.Wq.Out != 16 || attn.Wk.Out != 8 {
		t.Errorf("GQA widths: q out %d (want 16), k out %d (want 8)", attn.Wq.Out, attn.Wk.Out)
	}
}

func TestForwardOnLoadedWeights(t *testing.T) {
	dir := t.TempDir()
	cfg := tinyConfig()
	writeTinyCheckpoint(t, dir, cfg)

	gpt, err := FromDirectory(dir)
	if err != nil {
		t.Fatal(err)
	}

	ids := []int{1, 5, 2, 9, 9}
	logits := gpt.Forward(ids)

	if len(logits) != len(ids) {
		t.Fatalf("got %d logit rows, want %d", len(logits), len(ids))
	}
	for tPos, row := range logits {
		if len(row) != cfg.VocabSize {
			t.Fatalf("row %d: got %d logits, want %d", tPos, len(row), cfg.VocabSize)
		}
		for i, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("row %d index %d is %v — forward pass is numerically broken", tPos, i, v)
			}
		}
	}

	// Determinism: the same ids must produce bit-identical logits.
	again := gpt.Forward(ids)
	for tPos := range logits {
		for i := range logits[tPos] {
			if logits[tPos][i] != again[tPos][i] {
				t.Fatalf("forward is not deterministic at row %d index %d", tPos, i)
			}
		}
	}
}

// Causality is the one property we can assert without a reference
// implementation: truncating the input must not change the logits for the
// positions that remain, because no position may attend to the future.
func TestForwardIsCausal(t *testing.T) {
	dir := t.TempDir()
	writeTinyCheckpoint(t, dir, tinyConfig())

	gpt, err := FromDirectory(dir)
	if err != nil {
		t.Fatal(err)
	}

	full := gpt.Forward([]int{3, 7, 1, 4, 6})
	prefix := gpt.Forward([]int{3, 7, 1})

	for tPos := range prefix {
		for i := range prefix[tPos] {
			got, want := prefix[tPos][i], full[tPos][i]
			if math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("position %d index %d: prefix gave %v but full sequence gave %v — "+
					"later tokens are leaking into earlier positions", tPos, i, got, want)
			}
		}
	}
}

// A wrong HeadDim is the failure mode most likely to produce fluent nonsense
// instead of an error, so the loader must reject it at load time.
func TestLoaderRejectsWrongHeadDim(t *testing.T) {
	dir := t.TempDir()
	writeTinyCheckpoint(t, dir, tinyConfig())

	st, err := OpenSafetensorsDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	bad := tinyConfig()
	bad.HeadDim = 2 // the value NEmbed/NHead would have wrongly produced

	_, err = LoadQwen3(bad, st)
	if err == nil {
		t.Fatal("expected a shape error for a mismatched HeadDim")
	}
	if !strings.Contains(err.Error(), "q_proj") {
		t.Errorf("error should name the offending tensor, got: %v", err)
	}
}

func TestLoaderReportsAllMissingTensors(t *testing.T) {
	dir := t.TempDir()
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), []stTensor{
		{"model.embed_tokens.weight", "F32", []int{16, 8}, f32Bytes(ramp(128, 1)...)},
	})

	st, err := OpenSafetensorsDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, err = LoadQwen3(tinyConfig(), st)
	if err == nil {
		t.Fatal("expected errors for a checkpoint missing every layer tensor")
	}
	// Accumulating errors beats failing on the first one: you see the whole
	// list of what's wrong in a single run.
	if !strings.Contains(err.Error(), "model.norm.weight") {
		t.Errorf("expected the joined error to mention later tensors too, got: %v", err)
	}
}
