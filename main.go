package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
	"github.com/blaketylerfullerton/GoLlama/tools/tui"
	"github.com/blaketylerfullerton/GoLlama/tools/walkthrough"
)

// isTerminal reports whether f is attached to a terminal. Bubbletea needs one:
// under `go run . | less` or in CI there is nothing to draw on, so the splash is
// skipped rather than failing.
func isTerminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

// checkpointDir is where a real Qwen3 checkpoint is expected:
//
//	huggingface-cli download Qwen/Qwen3-0.6B --local-dir checkpoints/qwen3-0.6b
//
// When it's missing, main falls back to a tiny randomly initialized model so
// `go run .` still demonstrates every stage on a fresh clone.
const checkpointDir = "checkpoints/qwen3-0.6b"

// session is whichever model we managed to put together, plus a prompt to run
// through it.
type session struct {
	gpt    *model.GPT
	tok    *tokenizer.Tokenizer
	ids    []int
	prompt string
	dir    string // where the weights came from; empty for the random model
	real   bool
	// maxNewTokens stays small for the real model: at 0.6B with a naive matmul
	// each token is most of a second.
	maxNewTokens int
}

func main() {
	verbose := flag.Bool("v", false, "print the full layer-by-layer walkthrough")
	prompt := flag.String("prompt", "The capital of France is", "prompt to run through the model")
	tracePath := flag.String("trace", "", "write a trace of the forward pass to this file")
	noSplash := flag.Bool("no-splash", false, "skip the welcome screen and run straight away")
	traceChat := flag.Bool("trace-chat", false, "record what each reply attended to, for the past-conversations replay, and show live commit-layer depth in the chat footer; the inspect tools always trace, this only affects Chat (forces attention heads to run sequentially, which costs real throughput)")
	replayFile := flag.String("f", "", "replay a trace file in the inspect screen instead of running a model — needs no checkpoint")
	watermarkDemo := flag.Bool("watermark", false, "run a SynthID-Text-style watermarking demo: generate the prompt watermarked and plain with the same checkpoint, then run the detector on both")
	flag.Parse()

	// -f needs no checkpoint and none of the other screens: it opens straight
	// into the inspect screen on a trace already captured (by -trace, on an
	// earlier run), same as `inspect -f` used to before cmd/inspect merged
	// into this binary.
	if *replayFile != "" {
		if err := tui.StartInspectFile(*replayFile); err != nil {
			fmt.Fprintf(os.Stderr, "gollama: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// -watermark is its own printed walkthrough, same as -no-splash below,
	// just comparing two generations instead of narrating one.
	if *watermarkDemo {
		if err := runWatermarkDemo(*prompt); err != nil {
			fmt.Fprintf(os.Stderr, "gollama: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// The splash goes first, before any checkpoint is touched: it says what
	// hardware this is about to run on, then asks which tool to open and which
	// model to put on it, and shows what that costs in memory. Both are worth
	// knowing before a multi-second load. It's skipped when stdout isn't a
	// terminal so piping still works, and skipped with -no-splash, in which
	// case the default checkpoint is used if it's there.
	//
	// Every screen — splash, picker, about, history, chat, inspect — runs
	// inside this one call now, on one alternate screen. newChatEngine and
	// newInspectEngine are the only things tui can't do for itself: they're
	// what turn a typed line, or a prompt to trace, into a running model, and
	// package tui deliberately knows nothing about one.
	interactive := !*noSplash && isTerminal(os.Stdout)
	if interactive {
		if err := tui.Start(checkpointDir, *prompt, newChatEngine(*traceChat), newInspectEngine(), newWatermarkEngine()); err != nil {
			fmt.Fprintf(os.Stderr, "gollama: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// -no-splash and piped output keep the printed walkthrough, since there's
	// no screen to chat on and a script reading stdout wants the old fixed
	// output anyway.
	s, err := setup(checkpointDir, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gollama: %v\n", err)
		os.Exit(1)
	}
	if err := run(s, *verbose, *tracePath); err != nil {
		fmt.Fprintf(os.Stderr, "gollama: %v\n", err)
		os.Exit(1)
	}
}

func run(s *session, verbose bool, tracePath string) error {
	cfg := s.gpt.Config
	labels := make([]string, len(s.ids))
	for i, id := range s.ids {
		labels[i] = s.tok.Decode([]int{id})
	}

	// --- what we're running -------------------------------------------------
	if s.real {
		fmt.Printf("%s\n", s.dir)
	} else {
		fmt.Printf("no checkpoint at %s — using a tiny random model, so the numbers are noise\n"+
			"  huggingface-cli download Qwen/Qwen3-0.6B --local-dir %s\n", checkpointDir, checkpointDir)
	}
	fmt.Printf("%d layers · %d q heads / %d kv heads x %d dims · %s params\n",
		cfg.NLayer, cfg.NHead, cfg.NKVHead, cfg.HeadDim, count(paramTotal(cfg)))

	// --- tokenization -------------------------------------------------------
	fmt.Printf("\nprompt  %q\n", s.prompt)
	fmt.Printf("%d tokens  %s\n", len(s.ids), strings.Join(visible(labels), " | "))
	if !s.tok.PretokenizerIsExact() {
		fmt.Println("  (this tokenizer's pretokenizer pattern isn't one we recognise, so the" +
			" splits are approximate)")
	}

	// A tracer is only attached when someone is going to read the output. With
	// none, every hook in the forward pass is a no-op behind one nil check.
	var walk *walkthrough.Walkthrough
	tracers := []model.Tracer{}
	if verbose {
		walk = walkthrough.New(labels)
		tracers = append(tracers, walk)
	}
	traceWriter, closeTrace, err := openTrace(tracePath, s, labels)
	if err != nil {
		return err
	}
	defer closeTrace()
	if traceWriter != nil {
		tracers = append(tracers, traceWriter)
	}
	s.gpt.Trace = trace.Tee(tracers...)

	// --- the forward pass ---------------------------------------------------
	if verbose {
		fmt.Println(rule("forward pass, full detail for layer 0 head 0"))
	}
	start := time.Now()
	last, err := s.prefill()
	if err != nil {
		return err
	}
	prefillTime := time.Since(start)

	if verbose {
		fmt.Println(rule("where the parameters live"))
		printParamSplit(cfg)

		fmt.Println(rule("rotary position tables"))
		cos, sin := s.gpt.RotaryTables()
		walkthrough.PrintRotaryTable(cos, sin, 2)

		fmt.Println(rule("stage magnitudes (mean ‖x‖ over tokens)"))
		walk.PrintSummary()
	}

	// --- what it predicts ---------------------------------------------------
	fmt.Printf("\nnext token\n")
	for _, c := range model.TopCandidates(last, 1.0, 5) {
		fmt.Printf("  %5.1f%%  %q\n", 100*c.Prob, s.tok.Decode([]int{c.ID}))
	}
	if verbose {
		printTemperatures(s, last)
	}

	if traceWriter != nil {
		// Record the real output as one more lens readout, past the last block,
		// so the inspector has a row to compare intermediate layers against.
		traceWriter.LogitLens(cfg.NLayer, last, model.Argmax(last))
	}

	// --- generation ---------------------------------------------------------
	// Tracing comes off first, or it fires again for every generated token.
	s.gpt.Trace = nil

	// The prompt is printed unquoted so the streamed continuation reads on from
	// it as one sentence.
	fmt.Printf("\noutput  %s", s.prompt)
	genStart := time.Now()
	out, err := s.gpt.Generate(s.ids, model.GenerateOpts{
		MaxTokens:  s.maxNewTokens,
		SampleOpts: model.SampleOpts{Temperature: 0.7, TopK: 20, TopP: 0.95, Seed: 1},
		OnToken:    func(id int) { fmt.Print(s.tok.DecodeSkipSpecial([]int{id})) },
	})
	if err != nil {
		return err
	}
	genTime := time.Since(genStart)

	cache := model.NewKVCache(cfg)
	fmt.Printf("\n\nprefill %v · %d tokens in %v (%v/token) · kv cache %d KB/token\n",
		prefillTime.Round(time.Millisecond), len(out), genTime.Round(time.Millisecond),
		perToken(genTime, len(out)), cache.BytesPerToken()/1024)

	if traceWriter != nil {
		fmt.Printf("\nwrote %d trace events to %s\n", traceWriter.Events(), tracePath)
	}
	if !verbose {
		fmt.Println("\n-v for the layer-by-layer walkthrough · " +
			"go run . to pick a tool (Ablation, Attention, Attribution, Logit Lens) and explore interactively")
	}
	return nil
}

// prefill runs the prompt through the model and returns the final position's
// logits.
//
// It uses the cached path, which projects only the last row through the LM head
// instead of all of them. The uncached Forward is still used when tracing, since
// the walkthrough wants every position's intermediates.
func (s *session) prefill() ([]float32, error) {
	if s.gpt.Trace != nil {
		logits := s.gpt.Forward(s.ids)
		return logits[len(logits)-1], nil
	}
	return s.gpt.ForwardCached(s.ids, model.NewKVCache(s.gpt.Config))
}

func printTemperatures(s *session, last []float32) {
	fmt.Println(rule("temperature"))
	fmt.Println("Temperature divides the logits before softmax, so it changes how peaked the")
	fmt.Println("distribution is — it can never reorder the candidates.")
	for _, temp := range []float64{0.7, 1.0, 1.5} {
		fmt.Printf("\n  temperature %.1f\n", temp)
		for _, c := range model.TopCandidates(last, temp, 3) {
			fmt.Printf("    %6.2f%%  %q\n", 100*c.Prob, s.tok.Decode([]int{c.ID}))
		}
	}
}

// openTrace returns a trace writer plus a close function, or nils when no path
// was given. The writer also implements LogitLensTracer, which makes the engine
// project the residual stream through the LM head at every layer — so it only
// happens when something will read the result.
func openTrace(path string, s *session, labels []string) (*trace.Writer, func(), error) {
	if path == "" {
		return nil, func() {}, nil
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, func() {}, err
	}
	w, err := trace.NewWriter(f, traceHeader(s, labels), trace.Opts{
		Vocab:       func(id int) string { return s.tok.Decode([]int{id}) },
		Attribution: true,
	})
	if err != nil {
		_ = f.Close()
		return nil, func() {}, err
	}
	return w, func() {
		_ = w.Close()
		_ = f.Close()
	}, nil
}

// --- setup ------------------------------------------------------------------

// setup loads the checkpoint in dir, or the tiny random model when dir is empty
// or has no weights in it. An empty dir is a deliberate choice — the picker uses
// it for the built-in model — while a dir with nothing in it is the fresh-clone
// case, and both end up in the same place.
//
// An empty prompt is also a deliberate choice, made by the chat screen's engine:
// its prompts arrive one at a time off a channel, so there is nothing to encode
// up front. Only the printed walkthrough has a prompt to run at load time.
func setup(dir, prompt string) (*session, error) {
	if dir == "" {
		return setupDemo(prompt)
	}
	if model.HasWeights(dir) {
		return setupReal(dir, prompt)
	}
	return setupDemo(prompt)
}

func setupReal(dir, prompt string) (*session, error) {
	tok, err := tokenizer.FromDirectory(dir)
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}
	gpt, err := model.FromDirectory(dir)
	if err != nil {
		return nil, fmt.Errorf("loading model: %w", err)
	}
	ids, err := encodeOptional(tok, prompt)
	if err != nil {
		return nil, err
	}
	return &session{gpt: gpt, tok: tok, ids: ids, prompt: prompt, dir: dir,
		real: true, maxNewTokens: 3}, nil
}

func setupDemo(prompt string) (*session, error) {
	tok, err := tokenizer.FromDirectory("engine/tokenizer/testdata")
	if err != nil {
		return nil, fmt.Errorf("loading tokenizer: %w", err)
	}

	// Qwen3-shaped but tiny. Two properties are inherited from the real
	// Qwen3-0.6B deliberately: HeadDim is not NEmbed/NHead, and NKVHead <
	// NHead. So the shapes exercised here are the ones a real checkpoint hits.
	gpt, err := model.NewRandomGPT(model.GPTConfig{
		VocabSize:    tok.VocabSize(),
		NLayer:       2,
		NHead:        4,
		NKVHead:      2,
		NEmbed:       32,
		HeadDim:      16,
		Intermediate: 96,
		RopeBase:     1e6,
		NormEps:      1e-6,
		TieEmbed:     true,
		SequenceLen:  512,
	})
	if err != nil {
		return nil, err
	}
	ids, err := encodeOptional(tok, prompt)
	if err != nil {
		return nil, err
	}
	return &session{gpt: gpt, tok: tok, ids: ids, prompt: prompt,
		real: false, maxNewTokens: 12}, nil
}

// encodeOptional is encode for a session that may not have a prompt to run.
//
// The chat screen's engine is the caller with none: it loads a model so it can
// answer whatever gets typed later, and there is nothing to tokenize at that
// point. Without this, loading for a chat had to be handed some prompt, and the
// round-trip check below would then fail the whole load over a string that
// conversation was never going to run.
func encodeOptional(tok *tokenizer.Tokenizer, prompt string) ([]int, error) {
	if prompt == "" {
		return nil, nil
	}
	return encode(tok, prompt)
}

// encode tokenizes and checks the result round-trips, which catches a
// vocabulary mismatch straight away rather than as strange output later.
func encode(tok *tokenizer.Tokenizer, prompt string) ([]int, error) {
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		return nil, fmt.Errorf("prompt %q produced no tokens", prompt)
	}
	if got := tok.Decode(ids); got != prompt {
		return nil, fmt.Errorf("prompt does not round-trip: gave %q, got back %q", prompt, got)
	}
	return ids, nil
}

// traceHeader records what was run, so an inspector needs neither the model nor
// a tokenizer to make sense of the file.
func traceHeader(s *session, labels []string) trace.Header {
	cfg := s.gpt.Config
	tokens := make([]trace.Token, len(s.ids))
	for i, id := range s.ids {
		tokens[i] = trace.Token{ID: id, Text: labels[i]}
	}
	name := "random model (no checkpoint)"
	if s.real {
		name = s.dir
	}
	return trace.Header{
		Model:  name,
		Prompt: s.prompt,
		Tokens: tokens,
		Config: trace.ModelInfo{
			NLayer: cfg.NLayer, NEmbed: cfg.NEmbed,
			NHead: cfg.NHead, NKVHead: cfg.NKVHead,
			HeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize,
		},
	}
}

// --- formatting -------------------------------------------------------------

func rule(title string) string {
	return fmt.Sprintf("\n─── %s %s", title, strings.Repeat("─", max(0, 68-len(title))))
}

// visible makes a token's leading space apparent, since it's significant and
// otherwise invisible in a pipe-separated list.
func visible(labels []string) []string {
	out := make([]string, len(labels))
	for i, l := range labels {
		out[i] = walkthrough.Sanitize(l)
	}
	return out
}

// paramTotal counts the model's parameters. The split is worth knowing: at 0.6B
// the MLPs dominate, but the embedding table is still a quarter of the model
// because the vocabulary is so large.
func paramTotal(cfg model.GPTConfig) int {
	perLayer := cfg.NEmbed*cfg.QOut() + // q
		2*cfg.NEmbed*cfg.KVOut() + // k, v
		cfg.QOut()*cfg.NEmbed + // output projection
		2*cfg.HeadDim + // q_norm, k_norm
		3*cfg.NEmbed*cfg.Intermediate + // gate, up, down
		2*cfg.NEmbed // input + post-attention norms

	total := cfg.VocabSize*cfg.NEmbed + cfg.NLayer*perLayer + cfg.NEmbed
	if !cfg.TieEmbed {
		total += cfg.VocabSize * cfg.NEmbed
	}
	return total
}

// printParamSplit shows which parts of the model hold the weights. People are
// usually surprised that the embedding table is a quarter of a 0.6B model.
func printParamSplit(cfg model.GPTConfig) {
	attn := cfg.NLayer * (cfg.NEmbed*cfg.QOut() + 2*cfg.NEmbed*cfg.KVOut() +
		cfg.QOut()*cfg.NEmbed + 2*cfg.HeadDim)
	mlp := cfg.NLayer * 3 * cfg.NEmbed * cfg.Intermediate
	embed := cfg.VocabSize * cfg.NEmbed
	norms := cfg.NLayer*2*cfg.NEmbed + cfg.NEmbed
	total := paramTotal(cfg)

	for _, g := range []struct {
		name string
		n    int
	}{{"mlp", mlp}, {"attention", attn}, {"embeddings", embed}, {"norms", norms}} {
		fmt.Printf("  %-12s %12d  %5.1f%%\n", g.name, g.n, 100*float64(g.n)/float64(total))
	}
	fmt.Printf("  %-12s %12d\n", "total", total)
	if cfg.TieEmbed {
		fmt.Println("  embeddings are tied: the lm head reuses that table, no second copy")
	}
}

func count(n int) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%dM", n/1e6)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// perToken formats an average, guarding against a run that produced nothing.
func perToken(d time.Duration, n int) time.Duration {
	if n == 0 {
		return 0
	}
	return (d / time.Duration(n)).Round(time.Millisecond)
}
