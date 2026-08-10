package tui

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Arch is the handful of architecture numbers that decide what a checkpoint
// costs to run. It is deliberately not model.GPTConfig: the picker has to
// describe models that aren't on disk yet, so it needs numbers it can state
// without a loader, and tui stays free of any dependency on model/.
//
// HeadDim is stored rather than derived. On Qwen3 it is not NEmbed/NHead —
// 1024/16 is 64, but head_dim is 128 — and deriving it would understate the
// kv cache by half on exactly the model this repo ships with.
type Arch struct {
	NLayer       int
	NHead        int // query heads
	NKVHead      int // key/value heads; fewer than NHead means grouped-query attention
	HeadDim      int
	NEmbed       int
	Intermediate int
	VocabSize    int
	Context      int  // max_position_embeddings
	TieEmbed     bool // the lm head reuses the embedding table, so it costs nothing
}

// Params counts the weights.
//
// This mirrors paramTotal in main.go rather than calling it — package tui
// doesn't import model/, and the alternative is threading a config type through
// a screen whose whole job is to describe models it hasn't loaded. The formula
// is small and the two are checked against each other in the tests.
func (a Arch) Params() int64 {
	perLayer := int64(a.NEmbed)*int64(a.NHead*a.HeadDim) + // q
		2*int64(a.NEmbed)*int64(a.NKVHead*a.HeadDim) + // k, v
		int64(a.NHead*a.HeadDim)*int64(a.NEmbed) + // output projection
		2*int64(a.HeadDim) + // q_norm, k_norm
		3*int64(a.NEmbed)*int64(a.Intermediate) + // gate, up, down
		2*int64(a.NEmbed) // input + post-attention norms

	total := int64(a.VocabSize)*int64(a.NEmbed) + int64(a.NLayer)*perLayer + int64(a.NEmbed)
	if !a.TieEmbed {
		total += int64(a.VocabSize) * int64(a.NEmbed)
	}
	return total
}

// DiskBytes is what the weights take as bf16, which is how HuggingFace ships
// every Qwen3 checkpoint. Used to predict a download; for something already
// downloaded the real directory size is measured instead.
//
// The embedding table is counted twice under tied embeddings, because that is
// genuinely what gets downloaded: Qwen3 declares tie_word_embeddings and still
// ships an lm_head.weight identical to model.embed_tokens.weight. Predicting
// Params*2 would undershoot the real 0.6B download by 300MB. The loader throws
// that copy away, which is why ResidentBytes is not simply twice this.
func (a Arch) DiskBytes() int64 {
	n := a.Params()
	if a.TieEmbed {
		n += int64(a.VocabSize) * int64(a.NEmbed)
	}
	return n * 2
}

// ResidentBytes is what the weights take once loaded. The loader widens every
// stored bf16 to a float32, so memory is twice the download — the single
// biggest surprise on a small machine, and the reason this screen exists.
func (a Arch) ResidentBytes() int64 { return a.Params() * 4 }

// KVBytesPerToken is what each generated position adds to the cache: keys and
// values, for every layer and kv head, as float32.
func (a Arch) KVBytesPerToken() int64 {
	return int64(a.NLayer) * int64(a.NKVHead) * int64(a.HeadDim) * 2 * 4
}

// KVBytes is the cache at a given context length.
func (a Arch) KVBytes(tokens int) int64 { return a.KVBytesPerToken() * int64(tokens) }

// Model is one row in the picker.
type Model struct {
	Name string // "Qwen3-0.6B"
	Repo string // HuggingFace repo; empty for the built-in random model
	Dir  string // where the weights live, or would; empty for the random model
	Arch Arch
	// slug is the directory name under the checkpoint root. Dir is built from
	// it, rather than written out, so that Catalog's root argument actually
	// governs where everything is looked for — including in tests.
	slug string
	// Notes is the prose that shows up under the cursor, one paragraph per
	// entry. They're stored unwrapped so the panel can reflow them — the same
	// text has to read at 120 columns and at 60.
	Notes []string

	Installed bool  // model.safetensors is actually there
	OnDisk    int64 // measured size, meaningful only when Installed
	Custom    bool  // found on disk but not in the built-in catalog
	Demo      bool  // the tiny random model, which needs no download
}

// Download is the command that fetches this model.
func (m Model) Download() []string {
	if m.Repo == "" {
		return nil
	}
	return []string{
		"huggingface-cli download " + m.Repo,
		"  --local-dir " + m.Dir,
	}
}

// known is the Qwen3 dense family, smallest first.
//
// Only dense models are listed. The MoE checkpoints (30B-A3B and up) share the
// name but not the block: they route through experts that model/ has no code
// for, so offering them here would be offering a load that fails.
var known = []Model{
	{
		Name: "Qwen3-0.6B",
		Repo: "Qwen/Qwen3-0.6B",
		slug: "qwen3-0.6b",
		Arch: Arch{
			NLayer: 28, NHead: 16, NKVHead: 8, HeadDim: 128,
			NEmbed: 1024, Intermediate: 3072, VocabSize: 151936,
			Context: 40960, TieEmbed: true,
		},
		Notes: []string{
			"The one this repo is built around, and the only one small enough that a scalar Go matmul finishes a token while you're still looking at it.",
			"A quarter of the weights are the embedding table — 152k vocabulary times 1024 dims — and because the embeddings are tied, the lm head reuses it for free. Grouped-query attention halves the cache too: 16 query heads share 8 kv heads.",
		},
	},
	{
		Name: "Qwen3-1.7B",
		Repo: "Qwen/Qwen3-1.7B",
		slug: "qwen3-1.7b",
		Arch: Arch{
			NLayer: 28, NHead: 16, NKVHead: 8, HeadDim: 128,
			NEmbed: 2048, Intermediate: 6144, VocabSize: 151936,
			Context: 40960, TieEmbed: true,
		},
		Notes: []string{
			"The same 28 layers as 0.6B with everything twice as wide, which makes it a clean experiment: the shapes you already understand, four times the matmul. Noticeably more coherent, and noticeably slower for it.",
			"Note that the kv cache per token is identical to 0.6B's. The cache depends on layers and kv heads, and neither of those changed — only the weights and the per-token work grew.",
		},
	},
	{
		Name: "Qwen3-4B",
		Repo: "Qwen/Qwen3-4B",
		slug: "qwen3-4b",
		Arch: Arch{
			NLayer: 36, NHead: 32, NKVHead: 8, HeadDim: 128,
			NEmbed: 2560, Intermediate: 9728, VocabSize: 151936,
			Context: 40960, TieEmbed: true,
		},
		Notes: []string{
			"Where grouped-query attention starts paying real rent: 32 query heads over 8 kv heads, so the cache costs a quarter of what full multi-head would.",
			"36 layers of a 2560-wide stream. Widened to float32 the weights are past what most laptops want to hold, and every token is scalar Go over all of them. Practical to load; slow enough that you'll want the trace rather than a conversation.",
		},
	},
	{
		Name: "Qwen3-8B",
		Repo: "Qwen/Qwen3-8B",
		slug: "qwen3-8b",
		Arch: Arch{
			NLayer: 36, NHead: 32, NKVHead: 8, HeadDim: 128,
			NEmbed: 4096, Intermediate: 12288, VocabSize: 151936,
			// 8B is where Qwen3 stops tying the embeddings, so the lm head is a
			// second 622M-parameter table rather than an alias — worth 2.5GB of
			// resident float32 on its own.
			Context: 40960, TieEmbed: false,
		},
		Notes: []string{
			"Here for scale, not for use. Read the memory column on the right before pressing enter on this one — the weights are widened to float32 on load, so they want 30GB resident before a single token is generated.",
			"This is also where Qwen3 stops tying the embeddings: the lm head is a second 622M-parameter table rather than an alias to the first. Everything in model/ handles it, the shapes are the same shapes — but a naive matmul over 8B parameters takes seconds per token, not milliseconds.",
		},
	},
}

// demoModel is the fallback that makes a fresh clone work. It matches the tiny
// random config main.go builds when no checkpoint is present.
var demoModel = Model{
	Name: "tiny random model",
	Dir:  "",
	Arch: Arch{
		NLayer: 2, NHead: 4, NKVHead: 2, HeadDim: 16,
		NEmbed: 32, Intermediate: 96, VocabSize: 50257,
		Context: 512, TieEmbed: true,
	},
	Demo: true,
	Notes: []string{
		"No download, no weights — random numbers in the shape of a Qwen3. Every stage runs and every printout is real; only the predictions are noise. Start here if you want to watch the machinery rather than read the output.",
		"It's Qwen3-shaped where it matters: HeadDim is not NEmbed/NHead, and there are fewer kv heads than query heads. The awkward cases a real checkpoint hits are exercised here too.",
	},
}

// Catalog lists what you could run: the known Qwen3 models, marked with whether
// their weights are actually on disk, plus anything else found under root, plus
// the random model at the end.
//
// A checkpoint that's present overrides the catalog's guesses — its config.json
// is the truth about its own shape, and its directory is the truth about its
// size. The built-in numbers only fill in for models that haven't been
// downloaded yet, where there's nothing on disk to ask.
func Catalog(root string) []Model {
	out := make([]Model, 0, len(known)+2)
	claimed := map[string]bool{}

	for _, m := range known {
		m.Dir = filepath.Join(root, m.slug)
		if c := ScanCheckpoint(m.Dir); c.Present {
			m.Installed = true
			m.OnDisk = c.Bytes
			if a, ok := readArch(m.Dir); ok {
				m.Arch = a
			}
			claimed[filepath.Clean(m.Dir)] = true
		}
		out = append(out, m)
	}

	out = append(out, strays(root, claimed)...)
	return append(out, demoModel)
}

// strays finds checkpoints under root that aren't in the catalog — anything
// downloaded by hand, or a Qwen3 variant we don't ship a description for. They
// still load, so they still belong on the list; they just describe themselves
// entirely from their own config.json.
func strays(root string, claimed map[string]bool) []Model {
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []Model
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(root, e.Name())
		if claimed[filepath.Clean(dir)] {
			continue
		}
		c := ScanCheckpoint(dir)
		if !c.Present {
			continue
		}
		a, ok := readArch(dir)
		if !ok {
			continue
		}
		out = append(out, Model{
			Name:      e.Name(),
			Dir:       dir,
			Arch:      a,
			Installed: true,
			OnDisk:    c.Bytes,
			Custom:    true,
			Notes: []string{
				"Found in " + root + ", and not one of the checkpoints GoLlama ships a description for.",
				"Everything below and to the right is read out of its own config.json, so the numbers are exact even though there's nothing to say about the model itself. It'll load if the tensor names are Qwen3's.",
			},
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// hfArch mirrors the fields of HuggingFace's config.json that describe cost.
// It duplicates model/config.go's parsing on purpose: this screen runs before
// anything is loaded, and reading 700 bytes of JSON is the cheapest way to tell
// the truth about a checkpoint that's already sitting there.
type hfArch struct {
	VocabSize        int  `json:"vocab_size"`
	HiddenSize       int  `json:"hidden_size"`
	IntermediateSize int  `json:"intermediate_size"`
	NumHiddenLayers  int  `json:"num_hidden_layers"`
	NumAttnHeads     int  `json:"num_attention_heads"`
	NumKVHeads       int  `json:"num_key_value_heads"`
	HeadDim          int  `json:"head_dim"`
	MaxPositions     int  `json:"max_position_embeddings"`
	TieWordEmbed     bool `json:"tie_word_embeddings"`
}

// readArch reads a checkpoint's shape off disk. It reports false rather than an
// error: a directory that doesn't describe itself simply doesn't get listed, and
// there is nowhere on a splash screen to put a parse failure.
func readArch(dir string) (Arch, bool) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return Arch{}, false
	}
	var hf hfArch
	if err := json.Unmarshal(data, &hf); err != nil {
		return Arch{}, false
	}
	if hf.HiddenSize == 0 || hf.NumAttnHeads == 0 || hf.NumHiddenLayers == 0 {
		return Arch{}, false
	}
	a := Arch{
		NLayer: hf.NumHiddenLayers, NHead: hf.NumAttnHeads, NKVHead: hf.NumKVHeads,
		HeadDim: hf.HeadDim, NEmbed: hf.HiddenSize, Intermediate: hf.IntermediateSize,
		VocabSize: hf.VocabSize, Context: hf.MaxPositions, TieEmbed: hf.TieWordEmbed,
	}
	// The same fallbacks model/config.go applies, for the same reason: older
	// configs omit these and genuinely mean the conventional value.
	if a.HeadDim == 0 {
		a.HeadDim = a.NEmbed / a.NHead
	}
	if a.NKVHead == 0 {
		a.NKVHead = a.NHead
	}
	if a.Intermediate == 0 {
		a.Intermediate = 4 * a.NEmbed
	}
	return a, true
}

// Params formats a parameter count the way model cards do.
func params(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%dM", n/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%dK", n/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// padLeft right-aligns s in a field of n cells, for columns of numbers. It
// measures in cells rather than bytes: an em dash is one column and three
// bytes, and len() would shift the whole row it sits in.
func padLeft(s string, n int) string {
	w := lipgloss.Width(s)
	if w >= n {
		return s
	}
	return strings.Repeat(" ", n-w) + s
}
