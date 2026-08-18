package main

import (
	"context"
	"fmt"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
	"github.com/blaketylerfullerton/GoLlama/tools/tui"
)

// newInspectEngine is the inspect screen's counterpart to newChatEngine: it
// satisfies tui.InspectEngine, loading a checkpoint once and then serving
// requests off reqs until the screen is left. This used to be cmd/inspect's
// own program, with its own checkpoint loading; it now shares setup with
// chatEngine; the two only diverge in what they do with the loaded model,
// same as before.
func newInspectEngine() tui.InspectEngine {
	return func(ctx context.Context, dir string, reqs <-chan tui.InspectRequest, out chan<- tea.Msg) {
		inspectEngine(ctx, dir, reqs, out)
	}
}

func inspectEngine(ctx context.Context, dir string, reqs <-chan tui.InspectRequest, out chan<- tea.Msg) {
	defer close(out)

	s, err := setup(dir, "")
	if err != nil {
		emit(ctx, out, tui.InspectErr{Err: err})
		return
	}

	cfg := s.gpt.Config
	if !emit(ctx, out, tui.InspectReady{Info: trace.ModelInfo{
		NLayer: cfg.NLayer, NEmbed: cfg.NEmbed, NHead: cfg.NHead,
		NKVHead: cfg.NKVHead, HeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize,
	}}) {
		return
	}

	for {
		var req tui.InspectRequest
		select {
		case <-ctx.Done():
			return
		case req = <-reqs:
		}

		if req.Preview {
			if !emit(ctx, out, tui.InspectPreview{Tokens: previewTokens(s.tok, req.Prompt)}) {
				return
			}
			continue
		}

		if err := runInspectPrompt(s.gpt, s.tok, req, out); err != nil {
			emit(ctx, out, tui.InspectErr{Err: err})
			return
		}
		if !emit(ctx, out, tui.InspectRunDone{}) {
			return
		}
	}
}

// previewTokens is what answers a Preview request: the prompt's tokens,
// decoded and with whitespace made visible, since the caller has no
// tokenizer of its own to do that with.
func previewTokens(tok *tokenizer.Tokenizer, prompt string) []string {
	ids := tok.Encode(prompt)
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = sanitizeToken(tok.Decode([]int{id}))
	}
	return out
}

// sanitizeToken makes whitespace visible, the same convention tools/tui's own
// sanitize uses for token text — duplicated rather than exported, since this
// is the one place a real token's text is decoded outside that package.
func sanitizeToken(s string) string {
	out := make([]rune, 0, len(s))
	for _, r := range s {
		switch r {
		case '\n':
			out = append(out, '\\', 'n')
		case '\t':
			out = append(out, '\\', 't')
		case ' ':
			out = append(out, '_')
		default:
			out = append(out, r)
		}
	}
	return string(out)
}

// runInspectPrompt traces a prefill pass plus one pass per generated token,
// sending each as it completes so the UI fills in progressively — the same
// shape cmd/inspect's own runPrompt used.
func runInspectPrompt(gpt *model.GPT, tok *tokenizer.Tokenizer, req tui.InspectRequest, out chan<- tea.Msg) error {
	ids := tok.Encode(req.Prompt)
	if len(ids) == 0 {
		return fmt.Errorf("prompt %q produced no tokens", req.Prompt)
	}
	ablate := toModelHeadRefs(req.Ablate)

	cfg := gpt.Config
	header := trace.Header{
		Model:  "live",
		Prompt: req.Prompt,
		Tokens: inspectTokensOf(ids, tok),
		Config: trace.ModelInfo{
			NLayer: cfg.NLayer, NEmbed: cfg.NEmbed, NHead: cfg.NHead,
			NKVHead: cfg.NKVHead, HeadDim: cfg.HeadDim, VocabSize: cfg.VocabSize,
		},
	}
	collector := trace.NewCollector(header, trace.Opts{
		Vocab:       func(id int) string { return tok.Decode([]int{id}) },
		Attribution: true,
	})
	gpt.Trace = collector
	defer func() { gpt.Trace = nil }()

	// Prefill: the whole prompt in one pass. Attention here is the full lower
	// triangle, which is the view worth looking at.
	out <- statusMsg("prefill…") // best-effort; runPrompt's error path below is what's checked
	cache := model.NewKVCache(cfg)
	logits, err := gpt.ForwardCachedAblated(ids, cache, ablate)
	if err != nil {
		return err
	}
	collector.LogitLens(cfg.NLayer, logits, model.Argmax(logits)) // the real output, for comparison
	out <- tui.InspectStep{Label: "prefill", Tr: collector.Snapshot(), Ablated: len(ablate) > 0}

	// Decode: one token at a time, each with its own trace. That's what makes
	// "how did the model arrive at *this* token" a navigable axis.
	sampler := model.NewSampler(model.SampleOpts{}) // greedy, so runs reproduce
	stop := stopSet(cfg.EOSTokenIDs)
	seq := append([]int(nil), ids...)

	for n := 0; n < req.MaxTokens; n++ {
		next := sampler.Sample(logits)
		text := tok.Decode([]int{next})
		if stop[next] {
			break
		}

		out <- statusMsg(fmt.Sprintf("token %d/%d: %q…", n+1, req.MaxTokens, text))
		seq = append(seq, next)

		collector.Reset()
		// The token list grows so attention axes stay labelled as the sequence
		// extends.
		collector.Trace().Header.Tokens = inspectTokensOf(seq, tok)

		logits, err = gpt.ForwardCachedAblated([]int{next}, cache, ablate)
		if err != nil {
			return err
		}
		collector.LogitLens(cfg.NLayer, logits, model.Argmax(logits))
		out <- tui.InspectStep{Label: text, Tr: collector.Snapshot(), Ablated: len(ablate) > 0}
	}
	return nil
}

// statusMsg is a convenience alias so runInspectPrompt reads like
// cmd/inspect's own runPrompt did — it's just tui.InspectStatus.
func statusMsg(s string) tui.InspectStatus { return tui.InspectStatus(s) }

// toModelHeadRefs converts the tui-local HeadRef mirror back to the real
// model.HeadRef ForwardCachedAblated expects — the one place that conversion
// has to happen, since tools/tui itself never imports engine/model.
func toModelHeadRefs(hs []tui.HeadRef) []model.HeadRef {
	if len(hs) == 0 {
		return nil
	}
	out := make([]model.HeadRef, len(hs))
	for i, h := range hs {
		out[i] = model.HeadRef{Layer: h.Layer, Head: h.Head}
	}
	return out
}

func inspectTokensOf(ids []int, tok *tokenizer.Tokenizer) []trace.Token {
	out := make([]trace.Token, len(ids))
	for i, id := range ids {
		out[i] = trace.Token{ID: id, Text: tok.Decode([]int{id})}
	}
	return out
}
