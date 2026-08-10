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
	KindStage     Kind = "stage"      // the residual stream, or a branch off it
	KindAttention Kind = "attention"  // one head's causal attention weights
	KindRotary    Kind = "rotary"     // one head's q either side of the rotation
	KindNote      Kind = "note"       // commentary that isn't a tensor
	KindLogitLens Kind = "logit_lens" // what the model would predict at this depth
)

// FormatVersion is bumped when a change would break an existing reader.
const FormatVersion = 1

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

	// KindLogitLens
	Top []Candidate `json:"top,omitempty"`

	// KindNote
	Text string `json:"text,omitempty"`
}
