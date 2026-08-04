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

var specialTokens = []string{
	"<|bos|>",        // every document begins with the Beginning of Sequence (BOS) token that delimits documents
	"<|user_start|>", // user messages
	"<|user_end|>",
	"<|assistant_start|>", // assistant messages
	"<|assistant_end|>",
	"<|python_start|>", // assistant invokes python REPL tool
	"<|python_end|>",
	"<|output_start|>", // python REPL outputs back to assistant
	"<|output_end|>",
}

type Tokenizer struct {
	// Light wrapper around huggingface tokenzer for utils
	vocab     map[string]int //token string -> id
	idToToken []string       // id -> token string
	mergeRank map[pair]int   // (left, right) _> rank (lower = applied first)

	byteEnc map[byte]rune //raw byte -> printable rune
	byteDec map[rune]byte //printable rune -> raw byte

	pretokenRe *regexp.Regexp
}

type pair struct {
	left  string
	right string
}

type tokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
}

// encode converts text into token ids using byte level BPE.
// Same approach as GPT-2 stle tokenizer
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, chunk := range t.pretokenRe.FindAllString(text, -1) {
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

// VocabSize returns the number of tokens in the vocabulary, i.e. the
// number of rows an embedding table needs to cover every possible id.
func (t *Tokenizer) VocabSize() int {
	return len(t.idToToken)
}

func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.idToToken) {
			sb.WriteString(t.idToToken[id])
		}
	}

	// sb currenlty holds byte level runes. Map each back to its raw byte
	runes := []rune(sb.String())
	raw := make([]byte, len(runes))
	for i, r := range runes {
		raw[i] = t.byteDec[r]
	}
	return string(raw)
}

//----------------
// Constructors (Go doesnt have classmethods) so these are plain packages
// package level functions that return (*Tokenizer, error) instread of pythons (cls(..) pattern)
//

func FromPretrained(hfPath string) (*Tokenizer, error) {
	url := fmt.Sprint("https://huggingface.co/%s/resolve/main/tokenizer.json", hfPath)
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

	idToToken := make([]string, len(doc.Model.Vocab))
	for tok, id := range doc.Model.Vocab {
		if id < 0 || id >= len(idToToken) {
			return nil, fmt.Errorf("vocab id %d out of range for size %d", id, len(idToToken))
		}
		idToToken[id] = tok
	}

	mergeRank := make(map[pair]int, len(doc.Model.Merges))
	for i, m := range doc.Model.Merges {
		parts := strings.SplitN(m, " ", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("malformed merge entry %q", m)
		}
		mergeRank[pair{parts[0], parts[1]}] = i
	}

	byteEnc, byteDec := buildByteLevelAlphabet()

	return &Tokenizer{
		vocab:      doc.Model.Vocab,
		idToToken:  idToToken,
		mergeRank:  mergeRank,
		byteEnc:    byteEnc,
		byteDec:    byteDec,
		pretokenRe: pretokenRegexp(),
	}, nil
}

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

// Got this regex from claude
func pretokenRegexp() *regexp.Regexp {
	return regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\sa-zA-Z0-9]+|\s+`)
}
