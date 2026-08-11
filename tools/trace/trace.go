// Package trace records what a forward pass did, as a file.
//
// The point of writing to a file rather than driving a UI directly is
// decoupling: an inspector reads traces, so it can be developed and tested
// against fixtures, rewritten without touching the engine, and used to examine a
// run after the fact. The engine never learns that any of it exists — model/
// depends on nothing here, and the dependency arrow only ever points inward.
//
// The format is JSON Lines. The first line is a Header; every line after it is
// an Event. That makes traces greppable, streamable, and diffable, which matters
// more for this than compactness does.
package trace

// Kind identifies what an Event carries. Kinds are strings so an old trace
// stays readable after new ones are added.
type Kind string

const (
	KindStage       Kind = "stage"       // the residual stream, or a branch off it
	KindAttention   Kind = "attention"   // one head's causal attention weights
	KindRotary      Kind = "rotary"      // one head's q either side of the rotation
	KindNote        Kind = "note"        // commentary that isn't a tensor
	KindLogitLens   Kind = "logit_lens"  // what the model would predict at this depth
	KindAttribution Kind = "attribution" // how much one component moved the output
)

// FormatVersion is bumped when a change would break an existing reader.
//
// 2 added attribution events, and the target-token fields on a logit-lens
// readout. Both are additive, so a version-1 trace still reads — it simply has
// no events of the new kind.
const FormatVersion = 2

// Header is the first line of a trace: what was run, and on what.
type Header struct {
	Version int       `json:"version"`
	Model   string    `json:"model"`
	Prompt  string    `json:"prompt"`
	Tokens  []Token   `json:"tokens"`
	Config  ModelInfo `json:"config"`
}

// Token pairs an id with its text so a reader never needs a tokenizer.
type Token struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

// ModelInfo is the shape information an inspector needs. It deliberately
// mirrors rather than embeds model.GPTConfig, so the file format doesn't move
// every time the engine's config struct gains a field.
type ModelInfo struct {
	NLayer    int `json:"n_layer"`
	NEmbed    int `json:"n_embed"`
	NHead     int `json:"n_head"`
	NKVHead   int `json:"n_kv_head"`
	HeadDim   int `json:"head_dim"`
	VocabSize int `json:"vocab_size"`
}

// Candidate is one token and its probability.
type Candidate struct {
	ID   int     `json:"id"`
	Text string  `json:"text"`
	Prob float64 `json:"prob"`
}

// Effect is how much one component moved one token's logit. Positive pushes the
// token up, negative pushes it down, and the effects of every component on a
// token sum to that token's output logit.
type Effect struct {
	ID    int     `json:"id"`
	Text  string  `json:"text"`
	Logit float32 `json:"logit"`
}

// Component names which part of a layer an attribution event is about. Heads are
// numbered, so the Event's Head field carries the index and this says how to
// read it.
type Component string

const (
	ComponentHead  Component = "head"  // one attention head; Head is its index
	ComponentMLP   Component = "mlp"   // the layer's MLP
	ComponentEmbed Component = "embed" // the token embedding, at layer -1
)

// Event is one recorded moment. It's a single struct with omitted empties
// rather than a type per Kind: it keeps the file trivially readable and spares
// every consumer a type switch just to get at Layer and Kind.
type Event struct {
	Seq   int    `json:"seq"`
	Kind  Kind   `json:"kind"`
	Layer int    `json:"layer"` // -1 outside the block stack
	Name  string `json:"name,omitempty"`
	Head  int    `json:"head,omitempty"`

	// KindStage
	Tokens   int       `json:"tokens,omitempty"`
	Dims     int       `json:"dims,omitempty"`
	MeanNorm float64   `json:"mean_norm,omitempty"`
	Preview  []float32 `json:"preview,omitempty"` // leading dims of the last row

	// KindAttention. weights[i] has i+1 entries — everything past the diagonal
	// is masked and was never computed, so it isn't stored either.
	Weights [][]float32 `json:"weights,omitempty"`

	// KindRotary
	Before  []float32 `json:"before,omitempty"`
	After   []float32 `json:"after,omitempty"`
	NormIn  float64   `json:"norm_in,omitempty"`
	NormOut float64   `json:"norm_out,omitempty"`
	CosSim  float64   `json:"cos_sim,omitempty"`

	// KindLogitLens. Target is the token the pass finally predicted; Rank and
	// TargetProb are where it stood at this depth. Rank is 1-based, so zero
	// means the field wasn't recorded rather than "first".
	Top        []Candidate `json:"top,omitempty"`
	Entropy    float64     `json:"entropy,omitempty"` // nats, over the whole vocabulary
	TargetID   int         `json:"target_id,omitempty"`
	TargetText string      `json:"target_text,omitempty"`
	TargetRank int         `json:"target_rank,omitempty"`
	TargetProb float64     `json:"target_prob,omitempty"`

	// KindAttribution
	Component Component `json:"component,omitempty"`
	Norm      float64   `json:"norm,omitempty"` // length of this component's write
	Effects   []Effect  `json:"effects,omitempty"`

	// KindNote
	Text string `json:"text,omitempty"`
}
