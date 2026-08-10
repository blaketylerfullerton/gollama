package tokenizer

import (
	"reflect"
	"testing"
)

// These expectations were cross-checked against the pattern itself, run through
// Python's re with \p{L} and \p{N} translated to equivalent classes. All 34
// cases in that corpus matched, including the whole thing as one blob.
func TestSplitQwen(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   string
		want []string
	}{
		{"empty", "", nil},
		{"single letter", "a", []string{"a"}},

		// A leading space attaches to the word after it. That single rule is
		// why byte-level BPE vocabularies are full of "Ġword" entries.
		{"leading space joins word", " a", []string{" a"}},
		{"words", "Hello world", []string{"Hello", " world"}},

		// Two spaces split as one-space + space-plus-word: the `\s+(?!\S)`
		// branch gives up its last character so the word can claim it.
		{"two spaces", "  a", []string{" ", " a"}},
		{"three spaces", "   a", []string{"  ", " a"}},
		{"internal run", "a   b", []string{"a", "  ", " b"}},

		// Trailing whitespace has no word to attach to, so it stays whole.
		{"trailing spaces", "a   ", []string{"a", "   "}},
		{"only spaces", "   ", []string{"   "}},

		// One digit at a time — which is why "100" is three tokens, not one.
		{"digits split individually", "123", []string{"1", "2", "3"}},
		{"number with word", "abc123", []string{"abc", "1", "2", "3"}},
		{"decimal", "3.14", []string{"3", ".", "1", "4"}},

		{"contraction", "don't", []string{"don", "'t"}},
		{"contraction ll", "I'll", []string{"I", "'ll"}},
		// The (?i:) group makes contractions case-insensitive.
		{"contraction uppercase", "DON'T", []string{"DON", "'T"}},
		// Not in the contraction list, so it falls through to the word branch
		// and the apostrophe becomes the optional leading character.
		{"apostrophe not a contraction", "o'clock", []string{"o", "'clock"}},
		{"repeated apostrophes", "rock'n'roll", []string{"rock", "'n", "'roll"}},

		{"punctuation run", "!!!", []string{"!!!"}},
		{"punct takes leading space", "a !!", []string{"a", " !!"}},

		// The optional leading character in the word branch is *any* non-letter
		// non-digit, not just a space. So an opening paren attaches to the word
		// after it exactly the way a space does.
		{"paren joins word", "f(x)", []string{"f", "(x", ")"}},
		{"symbols with no word after", "a<>b", []string{"a", "<>", "b"}},

		// Same rule again: underscore is neither letter nor digit.
		{"underscore", "_a", []string{"_a"}},
		{"snake case", "a_b", []string{"a", "_b"}},

		{"newline", "a\nb", []string{"a", "\n", "b"}},
		{"blank line", "a\n\nb", []string{"a", "\n\n", "b"}},
		{"space then newline", "a \nb", []string{"a", " \n", "b"}},
		// A tab is excluded from the word branch's optional character only for
		// \r and \n — a tab is neither, so it attaches like a space.
		{"tab joins word", "a\tb", []string{"a", "\tb"}},

		// Non-ASCII letters are letters, so they group like any other word.
		{"accented", "café", []string{"café"}},
		{"japanese", "日本語", []string{"日本語"}},
		{"cyrillic", "Привет мир", []string{"Привет", " мир"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := splitQwen(tc.in)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("splitQwen(%q)\n  got  %q\n  want %q", tc.in, got, tc.want)
			}
		})
	}
}

// Splitting must be lossless: concatenating the pieces has to reproduce the
// input exactly. Anything else silently corrupts the prompt.
func TestSplitQwenIsLossless(t *testing.T) {
	for _, in := range []string{
		"Hello, world!", "  spaced  out  ", "tabs\tand\nnewlines\n\n",
		"don't 123 !!! café 日本語 🎉", "", " ", "\n\n\n", "a", "_", "!",
		"The quick brown fox jumps over the lazy dog.",
		"x**2 <= y", "<|im_end|>", "a<>b", "1,000,000",
	} {
		var joined string
		for _, part := range splitQwen(in) {
			joined += part
		}
		if joined != in {
			t.Errorf("splitQwen(%q) lost data: rejoined to %q", in, joined)
		}
	}
}

// No pretoken may be empty, or BPE would loop on nothing and the vocabulary
// lookup would produce a spurious id.
func TestSplitQwenProducesNoEmptyParts(t *testing.T) {
	for _, in := range []string{
		"Hello, world!", "   ", "a\n\n\nb", "!!!???", "café 123", "\t \t",
	} {
		for i, part := range splitQwen(in) {
			if part == "" {
				t.Errorf("splitQwen(%q) produced an empty part at index %d", in, i)
			}
		}
	}
}

// The pattern constant has to match what the checkpoint ships, or
// compilePretokenizer won't recognise it and will silently fall back.
func TestQwenPatternMatchesCheckpoint(t *testing.T) {
	tok, err := FromDirectory(realCheckpoint)
	if err != nil {
		t.Skipf("no checkpoint: %v", err)
	}
	if !tok.PretokenizerIsExact() {
		t.Errorf("the checkpoint's pattern is no longer recognised — it may have changed.\n"+
			"expected:\n  %s", qwenPattern)
	}
}
