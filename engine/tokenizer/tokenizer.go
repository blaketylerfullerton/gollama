package tokenizer

//package tokenizer impleents a dependamncy free byte level BPE
// Tokenizer simpoliar to tokenizers librarty

// This is from scratch using only go standard library
//Not a wire compabitable reimplementation but it can load save a compatible subsetr of tokenizer
// and encode / decode text the same general way GPT-2-style tokenizers do.

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

type Tokenizer struct {
	// Light wrapper around huggingface tokenzer for utils
	vocab     map[string]int //token string -> id
	idToToken []string       // id -> token string
	mergeRank map[pair]int   // (left, right) _> rank (lower = applied first)

	// special holds the ids of tokens from the added_tokens block — chat
	// markers like <|im_end|>, which live outside model.vocab and are never
	// produced by BPE merging.
	special map[int]bool

	byteEnc map[byte]rune //raw byte -> printable rune
	byteDec map[rune]byte //printable rune -> raw byte

	// split breaks text into pretokens; BPE merges only happen within one.
	split func(string) []string
	// pretokenExact records whether split reproduces the checkpoint's own
	// pretokenizer pattern. See compilePretokenizer.
	pretokenExact bool
}

type pair struct {
	left  string
	right string
}

type tokenizerJSON struct {
	// AddedTokens carries the special tokens. They are NOT in Model.Vocab and
	// their ids continue past it, so both have to be merged to get a complete
	// id -> token table. Qwen3 has 26 of them starting at 151643.
	AddedTokens  []addedToken    `json:"added_tokens"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	Model        struct {
		Vocab  map[string]int `json:"vocab"`
		Merges mergeList      `json:"merges"`
	} `json:"model"`
}

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

// mergeList decodes the BPE merge table, which the tokenizers library writes in
// two different shapes depending on its version:
//
//	older: ["Ġ Ġ", "i n", ...]        space-separated strings
//	newer: [["Ġ","Ġ"], ["i","n"], ...] explicit pairs
//
// Qwen3 ships the newer form. The older form can't be parsed unambiguously when
// a token itself contains a space, which is presumably why they changed it.
type mergeList [][2]string

func (m *mergeList) UnmarshalJSON(b []byte) error {
	var pairs [][]string
	if err := json.Unmarshal(b, &pairs); err == nil {
		out := make(mergeList, 0, len(pairs))
		for i, p := range pairs {
			if len(p) != 2 {
				return fmt.Errorf("merge %d has %d parts, want 2", i, len(p))
			}
			out = append(out, [2]string{p[0], p[1]})
		}
		*m = out
		return nil
	}

	var strs []string
	if err := json.Unmarshal(b, &strs); err != nil {
		return fmt.Errorf("merges is neither a list of pairs nor a list of strings")
	}
	out := make(mergeList, 0, len(strs))
	for i, s := range strs {
		left, right, ok := strings.Cut(s, " ")
		if !ok {
			return fmt.Errorf("malformed merge entry %d: %q", i, s)
		}
		out = append(out, [2]string{left, right})
	}
	*m = out
	return nil
}

// encode converts text into token ids using byte level BPE.
// Same approach as GPT-2 stle tokenizer
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, chunk := range t.split(text) {
		for _, sym := range t.bpe(chunk) {
			if id, ok := t.vocab[sym]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// bpe runs the merge loop on a single pretokenized chunk.
// Returning the final list of merged "symbol" strings. (before vocab lookup)
func (t *Tokenizer) bpe(chunk string) []string {
	// one symbol per raw byte, remapped thorugh the byte level alphabet
	symbols := make([]string, 0, len(chunk))
	for _, b := range []byte(chunk) {
		symbols = append(symbols, string(t.byteEnc[b]))
	}
	if len(symbols) < 2 {
		return symbols
	}

	for {
		// Find the adjacent pair with the lowest merge rank (applied first).
		bestRank, bestIdx := -1, -1
		for i := 0; i < len(symbols)-1; i++ {
			if rank, ok := t.mergeRank[pair{symbols[i], symbols[i+1]}]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank, bestIdx = rank, i
				}
			}
		}
		if bestIdx == -1 {
			break // no mergeable pair left
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx], append([]string{merged}, symbols[bestIdx+2:]...)...)
	}

	return symbols
}

// VocabSize returns the number of ids the tokenizer knows, counting the
// added_tokens block.
//
// This is not necessarily the model's vocab_size. Checkpoints often pad the
// embedding table for alignment — Qwen3-0.6B knows 151669 tokens but declares
// vocab_size 151936 — so the model can produce logits for ids that decode to
// nothing. Size embedding tables from config.json, not from this.
func (t *Tokenizer) VocabSize() int {
	return len(t.idToToken)
}

// Decode turns token ids back into text.
func (t *Tokenizer) Decode(ids []int) string { return t.decode(ids, false) }

// DecodeSkipSpecial is Decode with special tokens omitted — what you want when
// printing generated text, since nobody wants a literal <|im_end|> in output.
func (t *Tokenizer) DecodeSkipSpecial(ids []int) string { return t.decode(ids, true) }

// IsSpecial reports whether an id came from the added_tokens block.
func (t *Tokenizer) IsSpecial(id int) bool { return t.special[id] }

// decode walks the ids accumulating byte-level runes, translating them back to
// raw bytes in runs. Runs matter: a single UTF-8 character is often split
// across several tokens, so bytes can only be turned into a string in batches,
// not one token at a time.
//
// Special tokens are the exception. Their content is stored literally rather
// than byte-level encoded, so they break the run and pass straight through.
func (t *Tokenizer) decode(ids []int, skipSpecial bool) string {
	var out strings.Builder
	var pending []rune

	flush := func() {
		for _, r := range pending {
			if b, ok := t.byteDec[r]; ok {
				out.WriteByte(b)
			}
			// A rune outside the byte alphabet can't be mapped back; skipping
			// it beats emitting a NUL.
		}
		pending = pending[:0]
	}

	for _, id := range ids {
		if id < 0 || id >= len(t.idToToken) {
			continue // unknown id: the model can emit ids the tokenizer lacks
		}
		if t.special[id] {
			flush()
			if !skipSpecial {
				out.WriteString(t.idToToken[id])
			}
			continue
		}
		pending = append(pending, []rune(t.idToToken[id])...)
	}
	flush()
	return out.String()
}

//----------------
// Constructors (Go doesnt have classmethods) so these are plain packages
// package level functions that return (*Tokenizer, error) instread of pythons (cls(..) pattern)
//

func FromPretrained(hfPath string) (*Tokenizer, error) {
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/tokenizer.json", hfPath)
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("Downloadning tokenizer error for %q: %w", hfPath, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Downloading tokenikzer error")
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading tokenizer response for %q:", err)
	}
	return loadFromJSON(data)
}

// this function loads a tkenzier json frmo local dir
func FromDirectory(tokenizerDir string) (*Tokenizer, error) {
	tokenizerPath := filepath.Join(tokenizerDir, "tokenizer.json")
	data, error := os.ReadFile(tokenizerPath)
	if error != nil {
		return nil, fmt.Errorf("Reading %q: %w", tokenizerPath, error)
	}
	return loadFromJSON(data)
}

func loadFromJSON(data []byte) (*Tokenizer, error) {
	var doc tokenizerJSON
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("parsing tokenizer.json: %w", err)
	}
	if len(doc.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer.json has no vocab")
	}

	// Size the table by the highest id in either source, not by the number of
	// vocab entries: added_tokens continue past the end of model.vocab, and
	// sizing by len() alone silently drops every one of them.
	maxID := -1
	for _, id := range doc.Model.Vocab {
		if id < 0 {
			return nil, fmt.Errorf("vocab contains negative id %d", id)
		}
		if id > maxID {
			maxID = id
		}
	}
	for _, at := range doc.AddedTokens {
		if at.ID < 0 {
			return nil, fmt.Errorf("added token %q has negative id %d", at.Content, at.ID)
		}
		if at.ID > maxID {
			maxID = at.ID
		}
	}

	idToToken := make([]string, maxID+1)
	vocab := make(map[string]int, len(doc.Model.Vocab)+len(doc.AddedTokens))
	for tok, id := range doc.Model.Vocab {
		idToToken[id] = tok
		vocab[tok] = id
	}

	special := make(map[int]bool, len(doc.AddedTokens))
	for _, at := range doc.AddedTokens {
		idToToken[at.ID] = at.Content
		vocab[at.Content] = at.ID
		if at.Special {
			special[at.ID] = true
		}
	}

	mergeRank := make(map[pair]int, len(doc.Model.Merges))
	for i, m := range doc.Model.Merges {
		mergeRank[pair{m[0], m[1]}] = i
	}

	byteEnc, byteDec := buildByteLevelAlphabet()
	split, exact := compilePretokenizer(pretokenPatternFromJSON(doc.PreTokenizer))

	return &Tokenizer{
		vocab:         vocab,
		idToToken:     idToToken,
		mergeRank:     mergeRank,
		special:       special,
		byteEnc:       byteEnc,
		byteDec:       byteDec,
		split:         split,
		pretokenExact: exact,
	}, nil
}

// PretokenizerIsExact reports whether Encode uses the checkpoint's own
// pretokenizer pattern. When false, Encode is an approximation and its token
// boundaries will not always match HuggingFace — decode is unaffected.
func (t *Tokenizer) PretokenizerIsExact() bool { return t.pretokenExact }

// GPT-2s trick for makng BPE byte safe. Every byte 0-255 maps to a printable unicode rune
// Printable ascii map to themselves. the rest get remapped to runes
func buildByteLevelAlphabet() (enc map[byte]rune, dec map[rune]byte) {
	enc = make(map[byte]rune)
	dec = make(map[rune]byte)
	n := rune(0)
	bs := []int{}
	for _, r := range [][2]int{{'!', '~'}, {0xA1, 0xAC}, {0xAE, 0xFF}} {
		for b := r[0]; b <= r[1]; b++ {
			bs = append(bs, b)
		}
	}
	inBS := func(b int) bool {
		for _, x := range bs {
			if x == b {
				return true
			}
		}
		return false
	}
	for b := 0; b < 256; b++ {
		if inBS(b) {
			enc[byte(b)] = rune(b)
		} else {
			enc[byte(b)] = rune(256) + n
			n++
		}
	}
	for b, r := range enc {
		dec[r] = b
	}
	return
}

// pretokenPatternFromJSON digs the Split pattern out of a pre_tokenizer config,
// which is either a bare Split or a Sequence containing one.
func pretokenPatternFromJSON(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var node struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
		PreTokenizers []json.RawMessage `json:"pretokenizers"`
	}
	if err := json.Unmarshal(raw, &node); err != nil {
		return ""
	}
	if node.Pattern.Regex != "" {
		return node.Pattern.Regex
	}
	for _, child := range node.PreTokenizers {
		if p := pretokenPatternFromJSON(child); p != "" {
			return p
		}
	}
	return ""
}

// compilePretokenizer picks how to split text, given the pattern the checkpoint
// ships. Three cases, in order of preference:
//
//  1. The pattern is the known Qwen3/GPT-4 one, which splitQwen implements by
//     hand. Exact.
//  2. RE2 can compile it, so use the pattern directly. Exact.
//  3. Neither — fall back to an approximation and say so.
//
// Case 2 is rarer than it looks: every GPT-2-lineage pattern uses `\s+(?!\S)`,
// and Go's regexp is RE2, which rejects lookahead by design. Hence case 1
// existing at all.
func compilePretokenizer(pattern string) (split func(string) []string, exact bool) {
	switch {
	case pattern == qwenPattern:
		return splitQwen, true
	case pattern != "":
		if re, err := regexp.Compile(pattern); err == nil {
			return func(s string) []string { return re.FindAllString(s, -1) }, true
		}
	}
	// Unknown pattern that RE2 won't take. Qwen's hand-written splitter is
	// closer to any GPT-lineage tokenizer than the old approximation was, but
	// claiming exactness for a pattern we haven't read would be a lie.
	return splitQwen, false
}
