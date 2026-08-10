package tokenizer

import "unicode"

// qwenPattern is the pretokenizer regex Qwen3 (and every GPT-4-lineage
// tokenizer) ships in tokenizer.json:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)      contractions
//	[^\r\n\p{L}\p{N}]?\p{L}+          a word, optionally with one leading symbol
//	\p{N}                             one digit at a time
//	 ?[^\s\p{L}\p{N}]+[\r\n]*         punctuation run, optional leading space
//	\s*[\r\n]+                        whitespace ending in newlines
//	\s+(?!\S)                         whitespace, except the last char before a word
//	\s+                               any remaining whitespace
//
// Go's regexp is RE2 and rejects the `(?!\S)` lookahead outright, so this is
// implemented by hand below. splitQwen reproduces it exactly, including the
// backtracking behaviour the alternation depends on.
const qwenPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// splitQwen breaks text into pretokens. BPE merges only ever happen inside one
// of these, so getting the boundaries wrong changes the token ids even when the
// merge table is perfect.
func splitQwen(s string) []string {
	r := []rune(s)
	var out []string

	// Regex alternation takes the first branch that matches at each position,
	// so these are tried strictly in the pattern's order.
	matchers := []func([]rune, int) int{
		matchContraction,
		matchWord,
		matchDigit,
		matchPunctuation,
		matchNewlineRun,
		matchSpaceBeforeWord,
		matchSpaceRun,
	}

	for i := 0; i < len(r); {
		n := 0
		for _, match := range matchers {
			if n = match(r, i); n > 0 {
				break
			}
		}
		if n == 0 {
			// Unreachable: letters, digits, whitespace and everything else are
			// each covered by a branch. Advancing anyway beats looping forever.
			n = 1
		}
		out = append(out, string(r[i:i+n]))
		i += n
	}
	return out
}

// contractions are matched case-insensitively, per the (?i:) group. None is a
// prefix of another, so the order within the group doesn't affect the result.
var contractions = [][]rune{
	[]rune("'s"), []rune("'t"), []rune("'re"),
	[]rune("'ve"), []rune("'m"), []rune("'ll"), []rune("'d"),
}

func matchContraction(r []rune, i int) int {
	if r[i] != '\'' {
		return 0
	}
	for _, c := range contractions {
		if i+len(c) > len(r) {
			continue
		}
		ok := true
		for j := 1; j < len(c); j++ {
			if unicode.ToLower(r[i+j]) != c[j] {
				ok = false
				break
			}
		}
		if ok {
			return len(c)
		}
	}
	return 0
}

// matchWord implements `[^\r\n\p{L}\p{N}]?\p{L}+`: letters, optionally preceded
// by a single non-letter non-digit character that isn't a newline. That optional
// character is how a leading space ends up attached to the word after it, which
// is why " world" is one token rather than two.
func matchWord(r []rune, i int) int {
	// Try consuming the optional leading character first, as a greedy `?` does.
	if !isNewline(r[i]) && !unicode.IsLetter(r[i]) && !unicode.IsNumber(r[i]) {
		if end := runOfLetters(r, i+1); end > i+1 {
			return end - i
		}
		// No letters followed it, so the `?` gives the character back.
	}
	return runOfLetters(r, i) - i
}

// matchDigit implements `\p{N}` — exactly one digit. Numbers are deliberately
// split per digit, which is why "123" becomes three tokens.
func matchDigit(r []rune, i int) int {
	if unicode.IsNumber(r[i]) {
		return 1
	}
	return 0
}

// matchPunctuation implements ` ?[^\s\p{L}\p{N}]+[\r\n]*`.
func matchPunctuation(r []rune, i int) int {
	j := i
	if r[j] == ' ' {
		j++
	}
	k := j
	for k < len(r) && isSymbol(r[k]) {
		k++
	}
	if k == j {
		// Nothing matched the required `+`. Backtracking the optional space
		// can't help: the space itself isn't in the symbol class.
		return 0
	}
	for k < len(r) && isNewline(r[k]) {
		k++
	}
	return k - i
}

// matchNewlineRun implements `\s*[\r\n]+`: a whitespace run up to and including
// its last newline. The greedy `\s*` backtracks until a newline is available for
// the `+`, which lands the boundary right after the final newline.
func matchNewlineRun(r []rune, i int) int {
	end := runOfSpace(r, i)
	last := -1
	for j := i; j < end; j++ {
		if isNewline(r[j]) {
			last = j
		}
	}
	if last < 0 {
		return 0
	}
	return last + 1 - i
}

// matchSpaceBeforeWord implements `\s+(?!\S)`, the branch RE2 can't express.
//
// The lookahead means the match can't be followed by a non-space. Greedy `\s+`
// takes the whole run, and if a word follows it backtracks one character at a
// time until the character after the match is itself whitespace. The net effect
// is: give up the final space, so the word after it can claim that space via
// matchWord. That's what makes "  hello" split as " " + " hello".
func matchSpaceBeforeWord(r []rune, i int) int {
	end := runOfSpace(r, i)
	n := end - i
	switch {
	case n == 0:
		return 0
	case end == len(r):
		return n // nothing follows, so the lookahead is satisfied outright
	case n >= 2:
		return n - 1 // leave the last space for the following word
	default:
		return 0 // a lone space followed by a word: this branch can't match
	}
}

// matchSpaceRun implements the final `\s+` catch-all.
func matchSpaceRun(r []rune, i int) int {
	return runOfSpace(r, i) - i
}

func runOfLetters(r []rune, i int) int {
	for i < len(r) && unicode.IsLetter(r[i]) {
		i++
	}
	return i
}

func runOfSpace(r []rune, i int) int {
	for i < len(r) && unicode.IsSpace(r[i]) {
		i++
	}
	return i
}

// isSymbol is `[^\s\p{L}\p{N}]` — anything that isn't whitespace, a letter or a
// digit.
func isSymbol(c rune) bool {
	return !unicode.IsSpace(c) && !unicode.IsLetter(c) && !unicode.IsNumber(c)
}

func isNewline(c rune) bool { return c == '\r' || c == '\n' }
