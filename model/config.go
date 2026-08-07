package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// GPTConfig is every architectural knob the forward pass needs. Nothing in
// model/ is allowed to derive these from each other — HeadDim in particular
// is NOT NEmbed/NHead on Qwen3 (1024/16 = 64, but head_dim is 128), so
// deriving it silently corrupts every weight shape downstream.
type GPTConfig struct {
	VocabSize    int
	NLayer       int
	NHead        int // query heads
	NKVHead      int // key/value heads; < NHead means grouped-query attention
	NEmbed       int // hidden size / residual stream width
	HeadDim      int // width of one attention head
	Intermediate int // MLP hidden width (not necessarily 4*NEmbed)
	RopeBase     float64
	NormEps      float64
	TieEmbed     bool // lm_head shares weights with the embedding table
	SequenceLen  int
}

// QKVOut returns the output widths of the q and k/v projections. They differ
// under GQA: q is NHead wide, k and v are only NKVHead wide.
func (c GPTConfig) QOut() int  { return c.NHead * c.HeadDim }
func (c GPTConfig) KVOut() int { return c.NKVHead * c.HeadDim }

// GroupSize is how many query heads share a single kv head.
func (c GPTConfig) GroupSize() int { return c.NHead / c.NKVHead }

// hfConfig mirrors the subset of HuggingFace's config.json that we care about.
type hfConfig struct {
	VocabSize        int     `json:"vocab_size"`
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumHiddenLayers  int     `json:"num_hidden_layers"`
	NumAttnHeads     int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	RMSNormEps       float64 `json:"rms_norm_eps"`
	RopeTheta        float64 `json:"rope_theta"`
	TieWordEmbed     bool    `json:"tie_word_embeddings"`
	MaxPositions     int     `json:"max_position_embeddings"`
}

// ConfigFromDirectory reads a HuggingFace config.json and translates it into a
// GPTConfig. Fields HF omits fall back to the conventional defaults.
func ConfigFromDirectory(dir string) (GPTConfig, error) {
	path := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return GPTConfig{}, fmt.Errorf("reading %q: %w", path, err)
	}
	return ConfigFromJSON(data)
}

func ConfigFromJSON(data []byte) (GPTConfig, error) {
	var hf hfConfig
	if err := json.Unmarshal(data, &hf); err != nil {
		return GPTConfig{}, fmt.Errorf("parsing config.json: %w", err)
	}
	if hf.HiddenSize == 0 || hf.NumAttnHeads == 0 {
		return GPTConfig{}, fmt.Errorf("config.json missing hidden_size or num_attention_heads")
	}

	cfg := GPTConfig{
		VocabSize:    hf.VocabSize,
		NLayer:       hf.NumHiddenLayers,
		NHead:        hf.NumAttnHeads,
		NKVHead:      hf.NumKVHeads,
		NEmbed:       hf.HiddenSize,
		HeadDim:      hf.HeadDim,
		Intermediate: hf.IntermediateSize,
		RopeBase:     hf.RopeTheta,
		NormEps:      hf.RMSNormEps,
		TieEmbed:     hf.TieWordEmbed,
		SequenceLen:  hf.MaxPositions,
	}

	// Older configs omit head_dim and genuinely mean hidden/heads. Newer ones
	// (Qwen3) set it explicitly and it does not match — always prefer theirs.
	if cfg.HeadDim == 0 {
		cfg.HeadDim = cfg.NEmbed / cfg.NHead
	}
	if cfg.NKVHead == 0 {
		cfg.NKVHead = cfg.NHead // no GQA: every query head gets its own kv head
	}
	if cfg.NormEps == 0 {
		cfg.NormEps = 1e-6
	}
	if cfg.RopeBase == 0 {
		cfg.RopeBase = 10000
	}
	if cfg.Intermediate == 0 {
		cfg.Intermediate = 4 * cfg.NEmbed
	}

	return cfg, cfg.Validate()
}

// Validate catches the shape mistakes that would otherwise show up as
// plausible-looking garbage tokens rather than as an error.
func (c GPTConfig) Validate() error {
	if c.NKVHead == 0 || c.NHead%c.NKVHead != 0 {
		return fmt.Errorf("NHead (%d) must be a positive multiple of NKVHead (%d)", c.NHead, c.NKVHead)
	}
	if c.HeadDim%2 != 0 {
		return fmt.Errorf("HeadDim (%d) must be even for rotary to split it in half", c.HeadDim)
	}
	if c.NEmbed == 0 || c.NLayer == 0 || c.VocabSize == 0 {
		return fmt.Errorf("NEmbed, NLayer and VocabSize must all be set")
	}
	return nil
}
