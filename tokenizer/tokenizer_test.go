package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

// realCheckpoint is where the Qwen3 download lands. Tests that need it skip
// when it's absent, so the suite still passes on a fresh clone.
const realCheckpoint = "../checkpoints/qwen3-0.6b"

func writeTokenizerJSON(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// --- merges format compatibility --------------------------------------------

// The tokenizers library writes merges as space-separated strings in older
// versions and as explicit pairs in newer ones. Both must load.
func TestLoadsBothMergeFormats(t *testing.T) {
	const oldStyle = `{"model":{
		"vocab":{"a":0,"b":1,"ab":2},
		"merges":["a b"]}}`
	const newStyle = `{"model":{
		"vocab":{"a":0,"b":1,"ab":2},
		"merges":[["a","b"]]}}`

	for name, body := range map[string]string{"strings": oldStyle, "pairs": newStyle} {
		t.Run(name, func(t *testing.T) {
			tok, err := FromDirectory(writeTokenizerJSON(t, body))
			if err != nil {
				t.Fatal(err)
			}
			if len(tok.mergeRank) != 1 {
				t.Fatalf("got %d merges, want 1", len(tok.mergeRank))
			}
			if _, ok := tok.mergeRank[pair{"a", "b"}]; !ok {
				t.Errorf("merge (a,b) missing, got %v", tok.mergeRank)
			}
		})
	}
}

func TestRejectsMalformedMerges(t *testing.T) {
	// A pair with the wrong arity, and a string with no separator.
	for name, body := range map[string]string{
		"three-part pair": `{"model":{"vocab":{"a":0},"merges":[["a","b","c"]]}}`,
		"unsplittable":    `{"model":{"vocab":{"a":0},"merges":["ab"]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := FromDirectory(writeTokenizerJSON(t, body)); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

// --- added tokens -----------------------------------------------------------

// added_tokens live outside model.vocab and their ids continue past it. Sizing
// the id table by len(vocab) drops them all, which is silent: they decode to "".
func TestAddedTokensExtendTheIDTable(t *testing.T) {
	const body = `{
		"added_tokens":[
			{"id":3,"content":"<|im_end|>","special":true},
			{"id":4,"content":"<|pad|>","special":false}],
		"model":{"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}`

	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}

	if got := tok.VocabSize(); got != 5 {
		t.Errorf("VocabSize is %d, want 5 (3 vocab + 2 added)", got)
	}
	if got := tok.Decode([]int{3}); got != "<|im_end|>" {
		t.Errorf("decoding an added token gave %q, want %q", got, "<|im_end|>")
	}
	if !tok.IsSpecial(3) {
		t.Error("id 3 should be marked special")
	}
	if tok.IsSpecial(4) {
		t.Error(`id 4 has "special":false and should not be marked special`)
	}
	if tok.IsSpecial(0) {
		t.Error("a regular vocab token should not be marked special")
	}
}

func TestDecodeSkipSpecial(t *testing.T) {
	const body = `{
		"added_tokens":[{"id":3,"content":"<|im_end|>","special":true}],
		"model":{"vocab":{"hi":0,"!":1},"merges":[]}}`

	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{0, 3, 1}
	if got := tok.Decode(ids); got != "hi<|im_end|>!" {
		t.Errorf("Decode gave %q", got)
	}
	if got := tok.DecodeSkipSpecial(ids); got != "hi!" {
		t.Errorf("DecodeSkipSpecial gave %q, want %q", got, "hi!")
	}
}

func TestDecodeIgnoresUnknownIDs(t *testing.T) {
	const body = `{"model":{"vocab":{"a":0},"merges":[]}}`
	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}
	// A model can emit ids past the tokenizer's table (padded vocab_size).
	if got := tok.Decode([]int{0, 999, -1}); got != "a" {
		t.Errorf("got %q, want %q", got, "a")
	}
}

func TestRejectsNegativeIDs(t *testing.T) {
	const body = `{"model":{"vocab":{"a":-1},"merges":[]}}`
	if _, err := FromDirectory(writeTokenizerJSON(t, body)); err == nil {
		t.Error("expected an error for a negative vocab id")
	}
}

// --- byte-level alphabet ----------------------------------------------------

func TestByteLevelAlphabetRoundTrips(t *testing.T) {
	enc, dec := buildByteLevelAlphabet()
	if len(enc) != 256 {
		t.Fatalf("encoder covers %d bytes, want 256", len(enc))
	}
	if len(dec) != 256 {
		t.Fatalf("decoder covers %d runes, want 256 — the mapping is not injective", len(dec))
	}
	for b := 0; b < 256; b++ {
		if got := dec[enc[byte(b)]]; got != byte(b) {
			t.Fatalf("byte %d round-tripped to %d", b, got)
		}
	}
	// Printable ASCII maps to itself; a space does not (it becomes Ġ).
	if enc['A'] != 'A' {
		t.Errorf("printable ASCII should map to itself, got %q for 'A'", enc['A'])
	}
	if enc[' '] == ' ' {
		t.Error("space should be remapped out of the whitespace range")
	}
}

// Multi-byte UTF-8 is split across several tokens, so decoding has to
// reassemble bytes across token boundaries rather than per token.
func TestDecodeReassemblesMultibyteUTF8(t *testing.T) {
	enc, _ := buildByteLevelAlphabet()
	// "é" is 0xC3 0xA9 — two tokens, one character.
	var first, second string
	for _, b := range []byte("é") {
		if first == "" {
			first = string(enc[b])
		} else {
			second = string(enc[b])
		}
	}
	body := `{"model":{"vocab":{"` + first + `":0,"` + second + `":1},"merges":[]}}`

	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Decode([]int{0, 1}); got != "é" {
		t.Errorf("got %q, want %q", got, "é")
	}
}

// --- pretokenizer -----------------------------------------------------------

func TestPretokenizerExactWhenRE2CanCompile(t *testing.T) {
	const body = `{"model":{"vocab":{"a":0},"merges":[]},
		"pre_tokenizer":{"type":"Split","pattern":{"Regex":"\\p{L}+|\\s+"},
		"behavior":"Isolated"}}`

	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}
	if !tok.PretokenizerIsExact() {
		t.Error("a lookahead-free pattern should compile and be reported as exact")
	}
}

func TestPretokenizerFallsBackOnLookahead(t *testing.T) {
	// Go's regexp is RE2 and rejects (?!...). The loader must fall back rather
	// than fail, and must say so.
	const body = `{"model":{"vocab":{"a":0},"merges":[]},
		"pre_tokenizer":{"type":"Sequence","pretokenizers":[
			{"type":"Split","pattern":{"Regex":"\\s+(?!\\S)|\\s+"},"behavior":"Isolated"},
			{"type":"ByteLevel","add_prefix_space":false}]}}`

	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}
	if tok.PretokenizerIsExact() {
		t.Error("a pattern with negative lookahead cannot compile in RE2, so exact must be false")
	}
	if tok.split == nil {
		t.Error("fallback pattern should still be usable")
	}
}

func TestPretokenizerFallsBackWhenAbsent(t *testing.T) {
	tok, err := FromDirectory(writeTokenizerJSON(t, `{"model":{"vocab":{"a":0},"merges":[]}}`))
	if err != nil {
		t.Fatal(err)
	}
	if tok.PretokenizerIsExact() {
		t.Error("no pre_tokenizer in the file means we are not exact")
	}
}

// --- BPE encoding -----------------------------------------------------------

func TestBPEAppliesMergesByRank(t *testing.T) {
	enc, _ := buildByteLevelAlphabet()
	a, b, c := string(enc['a']), string(enc['b']), string(enc['c'])

	// Rank order decides which merge wins: (b,c) is rank 0 so it applies before
	// (a,b), which means "abc" becomes a + bc, not ab + c.
	body := `{"model":{"vocab":{
		"` + a + `":0,"` + b + `":1,"` + c + `":2,
		"` + b + c + `":3,"` + a + b + `":4},
		"merges":[["` + b + `","` + c + `"],["` + a + `","` + b + `"]]}}`

	tok, err := FromDirectory(writeTokenizerJSON(t, body))
	if err != nil {
		t.Fatal(err)
	}
	got := tok.Encode("abc")
	if len(got) != 2 || got[0] != 0 || got[1] != 3 {
		t.Errorf("got ids %v, want [0 3] (a + bc, because (b,c) outranks (a,b))", got)
	}
}

// --- the real checkpoint ----------------------------------------------------

// These run against the actual Qwen3 tokenizer.json when it's present, which is
// the only way to catch format drift in the file we don't control.
func TestRealQwen3Tokenizer(t *testing.T) {
	if _, err := os.Stat(filepath.Join(realCheckpoint, "tokenizer.json")); err != nil {
		t.Skipf("no checkpoint at %s", realCheckpoint)
	}

	tok, err := FromDirectory(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}

	// 151643 vocab entries (ids 0..151642) plus 26 added tokens through 151668.
	if got := tok.VocabSize(); got != 151669 {
		t.Errorf("VocabSize is %d, want 151669", got)
	}

	// The special tokens generation depends on must decode, not vanish.
	for id, want := range map[int]string{
		151643: "<|endoftext|>",
		151644: "<|im_start|>",
		151645: "<|im_end|>",
	} {
		if got := tok.Decode([]int{id}); got != want {
			t.Errorf("id %d decoded to %q, want %q", id, got, want)
		}
		if !tok.IsSpecial(id) {
			t.Errorf("id %d should be special", id)
		}
	}

	// Qwen3's pattern uses \s+(?!\S), which RE2 rejects — splitQwen implements
	// it by hand, so this is exact despite that.
	if !tok.PretokenizerIsExact() {
		t.Error("splitQwen should recognise and reproduce the Qwen3 pattern")
	}
}

// Encode against the real vocabulary, checked by id. These sequences were
// verified independently: they're the ones the model demonstrably responds to
// correctly ("The capital of France is" → " Paris").
func TestRealQwen3Encode(t *testing.T) {
	if _, err := os.Stat(filepath.Join(realCheckpoint, "tokenizer.json")); err != nil {
		t.Skipf("no checkpoint at %s", realCheckpoint)
	}
	tok, err := FromDirectory(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		text string
		want []int
	}{
		{"The capital of France is", []int{785, 6722, 315, 9625, 374}},
		{"1 2 3 4 5", []int{16, 220, 17, 220, 18, 220, 19, 220, 20}},
		{"Hello", []int{9707}},
	} {
		got := tok.Encode(tc.text)
		if len(got) != len(tc.want) {
			t.Errorf("Encode(%q) = %v, want %v", tc.text, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("Encode(%q) = %v, want %v", tc.text, got, tc.want)
				break
			}
		}
	}
}

// Encode/Decode must round-trip any input. This is the property that catches a
// pretokenizer which drops or duplicates characters.
func TestRealQwen3RoundTrip(t *testing.T) {
	if _, err := os.Stat(filepath.Join(realCheckpoint, "tokenizer.json")); err != nil {
		t.Skipf("no checkpoint at %s", realCheckpoint)
	}
	tok, err := FromDirectory(realCheckpoint)
	if err != nil {
		t.Fatal(err)
	}

	for _, text := range []string{
		"Hello, world!",
		"The capital of France is Paris.",
		"  leading and trailing   ",
		"multiple\n\nnewlines\n",
		"tabs\tand\tspaces",
		"CamelCase and snake_case and kebab-case",
		"don't can't we've I'll they're it's",
		"DON'T CAN'T WE'VE", // contractions are case-insensitive
		"numbers 1234567890 and 3.14159",
		"punctuation!!! ??? ...  ,,,",
		"unicode: café, naïve, 日本語, Привет, 🎉",
		"code: func main() { fmt.Println(\"hi\") }",
		"math: x**2 + y**2 <= z**2",
		"", // empty input must not panic
		" ",
		"\n",
		"a",
	} {
		ids := tok.Encode(text)
		if got := tok.Decode(ids); got != text {
			t.Errorf("round trip failed\n  in:  %q\n  out: %q\n  ids: %v", text, got, ids)
		}
	}
}
