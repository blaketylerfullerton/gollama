package main

import (
	"fmt"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/model"
	"github.com/blaketylerfullerton/GoLlama/tokenizer"
	"github.com/blaketylerfullerton/GoLlama/trace"
)

// Messages the engine goroutine sends to the UI.
type (
	// statusMsg is progress text shown while work is in flight.
	statusMsg string
	// readyMsg says the checkpoint is loaded and prompts can be submitted. It
	// carries the tokenizer so the UI can show tokenization as you type; that's
	// safe to share because nothing writes to it after loading.
	readyMsg struct {
		tok *tokenizer.Tokenizer
		cfg trace.ModelInfo
	}
	// stepMsg is one completed traced pass: the prefill, or one generated token.
	stepMsg struct {
		label string
		tr    *trace.Trace
	}
	// runDoneMsg says a whole prompt finished, so the UI can go back to
	// browsing.
	runDoneMsg struct{}
	doneMsg    struct{}
	errMsg     struct{ err error }
)

// request is one prompt to run.
type request struct {
	prompt    string
	maxTokens int
}

// waitFor turns the next message off a channel into a bubbletea command. It has
// to be re-issued after each message to keep listening — that's the standard way
// to stream from a goroutine into the update loop.
func waitFor(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return doneMsg{}
		}
		return msg
	}
}

// runEngine loads the checkpoint once, then serves prompts off reqs until it's
// closed. Loading is 1.5GB and a couple of seconds, so it must not happen per
// prompt.
func runEngine(checkpoint string, reqs <-chan request, out chan<- tea.Msg) {
	defer close(out)

	fail := func(err error) { out <- errMsg{err} }

	out <- statusMsg("loading tokenizer…")
	tok, err := tokenizer.FromDirectory(checkpoint)
	if err != nil {
		fail(fmt.Errorf("loading tokenizer: %w", err))
		return
	}

	out <- statusMsg("loading weights (1.5GB)…")
	gpt, err := model.FromDirectory(checkpoint)
	if err != nil {
		fail(fmt.Errorf("loading model: %w", err))
		return
	}

	cfg := gpt.Config
	out <- readyMsg{tok: tok, cfg: trace.ModelInfo{
		NLayer: cfg.NLayer, NEmbed: cfg.NEmbed, NHead: cfg.NHead,
		NKVHead: cfg.NKVHead, HeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize,
	}}

	for req := range reqs {
		if err := runPrompt(gpt, tok, req, out); err != nil {
			fail(err)
			return
		}
		out <- runDoneMsg{}
	}
}

// runPrompt traces a prefill pass plus one pass per generated token, sending
// each as it completes so the UI fills in progressively.
func runPrompt(gpt *model.GPT, tok *tokenizer.Tokenizer, req request, out chan<- tea.Msg) error {
	ids := tok.Encode(req.prompt)
	if len(ids) == 0 {
		return fmt.Errorf("prompt %q produced no tokens", req.prompt)
	}

	cfg := gpt.Config
	header := trace.Header{
		Model:  "live",
		Prompt: req.prompt,
		Tokens: tokensOf(ids, tok),
		Config: trace.ModelInfo{
			NLayer: cfg.NLayer, NEmbed: cfg.NEmbed, NHead: cfg.NHead,
			NKVHead: cfg.NKVHead, HeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize,
		},
	}
	collector := trace.NewCollector(header, trace.Opts{
		Vocab: func(id int) string { return tok.Decode([]int{id}) },
	})
	gpt.Trace = collector
	defer func() { gpt.Trace = nil }()

	// Prefill: the whole prompt in one pass. Attention here is the full lower
	// triangle, which is the view worth looking at.
	out <- statusMsg("prefill…")
	cache := model.NewKVCache(cfg)
	logits, err := gpt.ForwardCached(ids, cache)
	if err != nil {
		return err
	}
	collector.LogitLens(cfg.NLayer, logits) // the real output, for comparison
	out <- stepMsg{label: "prefill", tr: collector.Snapshot()}

	// Decode: one token at a time, each with its own trace. That's what makes
	// "how did the model arrive at *this* token" a navigable axis.
	sampler := model.NewSampler(model.SampleOpts{}) // greedy, so runs reproduce
	seq := append([]int(nil), ids...)

	for n := 0; n < req.maxTokens; n++ {
		next := sampler.Sample(logits)
		text := tok.Decode([]int{next})
		if isStop(next, cfg.EOSTokenIDs) {
			break
		}

		out <- statusMsg(fmt.Sprintf("token %d/%d: %q…", n+1, req.maxTokens, text))
		seq = append(seq, next)

		collector.Reset()
		// The token list grows so attention axes stay labelled as the sequence
		// extends.
		collector.Trace().Header.Tokens = tokensOf(seq, tok)

		logits, err = gpt.ForwardCached([]int{next}, cache)
		if err != nil {
			return err
		}
		collector.LogitLens(cfg.NLayer, logits)
		out <- stepMsg{label: text, tr: collector.Snapshot()}
	}
	return nil
}

func tokensOf(ids []int, tok *tokenizer.Tokenizer) []trace.Token {
	out := make([]trace.Token, len(ids))
	for i, id := range ids {
		out[i] = trace.Token{ID: id, Text: tok.Decode([]int{id})}
	}
	return out
}

func isStop(id int, stops []int) bool {
	for _, s := range stops {
		if id == s {
			return true
		}
	}
	return false
}
