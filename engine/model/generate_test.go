package model

import (
	"testing"
)

func tinyGPT(t *testing.T) *GPT {
	t.Helper()
	dir := t.TempDir()
	cfg := tinyConfig()
	writeTinyCheckpoint(t, dir, cfg)
	gpt, err := FromDirectory(dir)
	if err != nil {
		t.Fatal(err)
	}
	return gpt
}

func TestGenerateRespectsMaxTokens(t *testing.T) {
	gpt := tinyGPT(t)
	for _, n := range []int{0, 1, 5} {
		got, err := gpt.Generate([]int{1, 2}, GenerateOpts{MaxTokens: n})
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != n {
			t.Errorf("MaxTokens %d: got %d tokens", n, len(got))
		}
	}
}

func TestGenerateIsDeterministicUnderSeed(t *testing.T) {
	gpt := tinyGPT(t)
	opts := GenerateOpts{MaxTokens: 8, SampleOpts: SampleOpts{Temperature: 1, Seed: 5}}

	a, err := gpt.Generate([]int{3, 1}, opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := gpt.Generate([]int{3, 1}, opts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed diverged at token %d: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestGenerateGreedyIsRepeatable(t *testing.T) {
	gpt := tinyGPT(t)
	opts := GenerateOpts{MaxTokens: 6} // Temperature 0 => greedy

	a, err := gpt.Generate([]int{2}, opts)
	if err != nil {
		t.Fatal(err)
	}
	b, err := gpt.Generate([]int{2}, opts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("greedy decoding is not repeatable at token %d: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestGenerateDoesNotMutatePrompt(t *testing.T) {
	gpt := tinyGPT(t)
	prompt := []int{4, 5, 6}
	// Extra capacity is the dangerous case: a naive append would write into
	// the caller's backing array instead of copying.
	padded := make([]int, 3, 32)
	copy(padded, prompt)

	if _, err := gpt.Generate(padded, GenerateOpts{MaxTokens: 5}); err != nil {
		t.Fatal(err)
	}
	for i := range prompt {
		if padded[i] != prompt[i] {
			t.Errorf("prompt was mutated at %d: got %d, want %d", i, padded[i], prompt[i])
		}
	}
	if len(padded) != 3 {
		t.Errorf("prompt length changed to %d", len(padded))
	}
}

func TestGenerateStopsOnStopToken(t *testing.T) {
	gpt := tinyGPT(t)

	// Greedy decoding is deterministic, so find what it would produce first,
	// then declare that token a stop token and expect nothing at all.
	first, err := gpt.Generate([]int{1, 1}, GenerateOpts{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 1 {
		t.Fatalf("setup: expected 1 token, got %d", len(first))
	}

	got, err := gpt.Generate([]int{1, 1}, GenerateOpts{MaxTokens: 10, Stop: []int{first[0]}})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want no tokens — the first sampled token was a stop token", got)
	}
}

func TestGenerateHonorsConfigEOS(t *testing.T) {
	gpt := tinyGPT(t)

	first, err := gpt.Generate([]int{1, 1}, GenerateOpts{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	// Same as above, but via the checkpoint's eos_token_id rather than opts.
	gpt.Config.EOSTokenIDs = []int{first[0]}

	got, err := gpt.Generate([]int{1, 1}, GenerateOpts{MaxTokens: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want no tokens — config EOS should stop generation", got)
	}
}

func TestGenerateStreamsEveryToken(t *testing.T) {
	gpt := tinyGPT(t)

	var streamed []int
	got, err := gpt.Generate([]int{1, 2}, GenerateOpts{
		MaxTokens:  6,
		SampleOpts: SampleOpts{Temperature: 1, Seed: 2},
		OnToken:    func(id int) { streamed = append(streamed, id) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(streamed) != len(got) {
		t.Fatalf("streamed %d tokens but returned %d", len(streamed), len(got))
	}
	for i := range got {
		if streamed[i] != got[i] {
			t.Errorf("token %d: streamed %d, returned %d", i, streamed[i], got[i])
		}
	}
}

func TestGenerateStopsAtContextLimit(t *testing.T) {
	gpt := tinyGPT(t)
	gpt.Config.SequenceLen = 5

	got, err := gpt.Generate([]int{1, 2, 3}, GenerateOpts{MaxTokens: 100})
	if err != nil {
		t.Fatal(err)
	}
	// Prompt of 3 with a limit of 5 leaves room for exactly 2 more.
	if len(got) != 2 {
		t.Errorf("got %d tokens, want 2 before hitting the context limit", len(got))
	}
}

func TestGenerateRejectsBadInput(t *testing.T) {
	gpt := tinyGPT(t)

	if _, err := gpt.Generate(nil, GenerateOpts{MaxTokens: 4}); err == nil {
		t.Error("expected an error for an empty prompt")
	}
	if _, err := gpt.Generate([]int{1}, GenerateOpts{MaxTokens: -1}); err == nil {
		t.Error("expected an error for negative MaxTokens")
	}
	// Out-of-range ids would otherwise panic inside the embedding lookup.
	if _, err := gpt.Generate([]int{99999}, GenerateOpts{MaxTokens: 1}); err == nil {
		t.Error("expected an error for a token id outside the vocab")
	}
}

func TestConfigParsesEOSAsScalarOrList(t *testing.T) {
	scalar := `{"hidden_size":8,"num_attention_heads":4,"num_hidden_layers":1,
		"vocab_size":16,"eos_token_id":151645}`
	cfg, err := ConfigFromJSON([]byte(scalar))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.EOSTokenIDs) != 1 || cfg.EOSTokenIDs[0] != 151645 {
		t.Errorf("got %v, want [151645]", cfg.EOSTokenIDs)
	}

	list := `{"hidden_size":8,"num_attention_heads":4,"num_hidden_layers":1,
		"vocab_size":16,"eos_token_id":[151645,151643]}`
	cfg, err = ConfigFromJSON([]byte(list))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.EOSTokenIDs) != 2 {
		t.Errorf("got %v, want two ids", cfg.EOSTokenIDs)
	}

	absent := `{"hidden_size":8,"num_attention_heads":4,"num_hidden_layers":1,"vocab_size":16}`
	cfg, err = ConfigFromJSON([]byte(absent))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.EOSTokenIDs) != 0 {
		t.Errorf("got %v, want empty when eos_token_id is absent", cfg.EOSTokenIDs)
	}
}
