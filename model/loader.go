package model

import (
	"errors"
	"fmt"
)

// FromDirectory loads a HuggingFace Qwen3 checkpoint: config.json for the
// architecture, model.safetensors (possibly sharded) for the weights.
func FromDirectory(dir string) (*GPT, error) {
	cfg, err := ConfigFromDirectory(dir)
	if err != nil {
		return nil, err
	}
	st, err := OpenSafetensorsDir(dir)
	if err != nil {
		return nil, err
	}
	return LoadQwen3(cfg, st)
}

// loader accumulates errors so the caller gets every shape mismatch in one
// pass instead of playing whack-a-mole one panic at a time.
type loader struct {
	st   *Safetensors
	errs []error
}

// tensor fetches a tensor and asserts its shape. The shape check is the whole
// point: a wrong HeadDim or a GQA mixup produces tensors that load fine and
// generate fluent nonsense, so it has to fail loudly here instead.
func (l *loader) tensor(name string, want ...int) []float32 {
	got, err := l.st.Shape(name)
	if err != nil {
		l.errs = append(l.errs, err)
		return nil
	}
	if len(got) != len(want) {
		l.errs = append(l.errs, fmt.Errorf("tensor %q: want %d dims %v, checkpoint has %d dims %v",
			name, len(want), want, len(got), got))
		return nil
	}
	for i := range want {
		if got[i] != want[i] {
			l.errs = append(l.errs, fmt.Errorf("tensor %q: want shape %v, checkpoint has %v", name, want, got))
			return nil
		}
	}
	data, err := l.st.Tensor(name)
	if err != nil {
		l.errs = append(l.errs, err)
		return nil
	}
	return data
}

// linear loads a weight stored as (out, in) row-major. PyTorch's nn.Linear
// already uses that layout, so unlike GPT-2's Conv1D there is no transpose.
// Qwen3 has no biases anywhere, hence no bias argument.
func (l *loader) linear(name string, out, in int) Linear {
	return Linear{Weight: l.tensor(name, out, in), In: in, Out: out}
}

func (l *loader) norm(name string, dim int, eps float64) RMSNorm {
	return RMSNorm{Weight: l.tensor(name, dim), Eps: eps}
}

// LoadQwen3 maps HuggingFace's Qwen3 tensor names onto our structs.
func LoadQwen3(cfg GPTConfig, st *Safetensors) (*GPT, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	l := &loader{st: st}

	wte := l.tensor("model.embed_tokens.weight", cfg.VocabSize, cfg.NEmbed)

	blocks := make([]Block, cfg.NLayer)
	for i := range blocks {
		p := fmt.Sprintf("model.layers.%d.", i)
		blocks[i] = Block{
			InputNorm: l.norm(p+"input_layernorm.weight", cfg.NEmbed, cfg.NormEps),
			Attn: Attention{
				Wq: l.linear(p+"self_attn.q_proj.weight", cfg.QOut(), cfg.NEmbed),
				Wk: l.linear(p+"self_attn.k_proj.weight", cfg.KVOut(), cfg.NEmbed),
				Wv: l.linear(p+"self_attn.v_proj.weight", cfg.KVOut(), cfg.NEmbed),
				Wo: l.linear(p+"self_attn.o_proj.weight", cfg.NEmbed, cfg.QOut()),
				// QK-norm weights are per head_dim and shared across heads.
				QNorm:   l.norm(p+"self_attn.q_norm.weight", cfg.HeadDim, cfg.NormEps),
				KNorm:   l.norm(p+"self_attn.k_norm.weight", cfg.HeadDim, cfg.NormEps),
				NHead:   cfg.NHead,
				NKVHead: cfg.NKVHead,
				HeadDim: cfg.HeadDim,
			},
			PostAttnNorm: l.norm(p+"post_attention_layernorm.weight", cfg.NEmbed, cfg.NormEps),
			MLP: MLP{
				Gate: l.linear(p+"mlp.gate_proj.weight", cfg.Intermediate, cfg.NEmbed),
				Up:   l.linear(p+"mlp.up_proj.weight", cfg.Intermediate, cfg.NEmbed),
				Down: l.linear(p+"mlp.down_proj.weight", cfg.NEmbed, cfg.Intermediate),
			},
		}
	}

	// Tied embeddings: small Qwen3 models omit lm_head.weight entirely and
	// reuse the embedding table. Trust the checkpoint over the config flag —
	// if the tensor is absent, tying is the only option regardless.
	var lmHead Linear
	if st.Has("lm_head.weight") {
		lmHead = l.linear("lm_head.weight", cfg.VocabSize, cfg.NEmbed)
	} else {
		lmHead = Linear{Weight: wte, In: cfg.NEmbed, Out: cfg.VocabSize}
	}

	gpt := &GPT{
		Config:    cfg,
		WTE:       wte,
		Blocks:    blocks,
		FinalNorm: l.norm("model.norm.weight", cfg.NEmbed, cfg.NormEps),
		LMHead:    lmHead,
	}

	if len(l.errs) > 0 {
		return nil, fmt.Errorf("loading checkpoint: %w", errors.Join(l.errs...))
	}
	return gpt, nil
}
