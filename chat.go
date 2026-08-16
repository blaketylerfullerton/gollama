package main

import (
	"context"
	"fmt"
	"sort"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
	"github.com/blaketylerfullerton/GoLlama/tools/tui"
)

// chatMaxTokens caps a single reply. The real model is a scalar matmul in Go —
// hundreds of milliseconds a token — so a reply has to stay short enough that
// pressing enter doesn't feel like it hung. The random model costs nothing per
// token, so it can afford to run on longer.
func chatMaxTokens(real bool) int {
	if real {
		return 48
	}
	return 96
}

// chatCandidates is how many entries go in each ranked list on the inspect
// tab. More than a handful stops being scannable, which is the only reason to
// show a list instead of the raw distribution.
const chatCandidates = 4

// chatEngine is what the chat screen talks to: it loads the checkpoint, then
// turns typed lines into generated ones. This is the one thing package tui
// deliberately doesn't know how to do — the screen streams strings and ranked
// lists, and this is where those come from a real *model.GPT (or the random
// one) instead. It satisfies tui.Engine, and tui.Start is handed it by main.
//
// It runs for the lifetime of one chat screen, holding one KV cache across every
// turn: what you type second is a continuation of what the model already said
// after the first, not a fresh prompt, so the cache has to carry the whole
// conversation rather than restart it.
//
// The checkpoint is loaded in here rather than before the screen opens, and
// that's deliberate: the picker already knows the model's name and shape from
// its catalog, so chat can be on screen saying "loading…" while this does the
// multi-second read behind it.
//
// ctx ends the whole thing. It's cancelled when the user leaves the chat screen,
// which is what stops this goroutine holding a checkpoint resident for the rest
// of the program's life — every model picked would otherwise leave one behind.
//
// It also traces its own decode step, one token at a time, purely so each turn
// records what a token attended to and what the model ranked as likely to follow
// it — the history screen steps back through that later. That's the same
// instrumentation cmd/inspect runs, just read out as two short lists instead of
// full matrices.
//
// That instrumentation isn't free: tracing forces every attention head to run
// sequentially instead of across goroutines (see the comment on Attention.Forward),
// which costs an order of magnitude of throughput on the real model. newChatEngine
// takes that trade-off as a parameter rather than always paying it — traceAttention
// defaults to off (see -trace-chat), which still streams tokens and next-token
// candidates, it just leaves the "what did it attend to" list empty.
func newChatEngine(traceAttention bool) tui.Engine {
	return func(ctx context.Context, dir string, reqs <-chan string, out chan<- tea.Msg) {
		chatEngine(ctx, dir, reqs, out, traceAttention)
	}
}

func chatEngine(ctx context.Context, dir string, reqs <-chan string, out chan<- tea.Msg, traceAttention bool) {
	defer close(out)

	// No status is sent before the checkpoint is loaded: tui.Chat already opens
	// in its loading phase showing "loading…", and any ChatStatus — even one
	// that just repeats that same text — flips it straight to idle. Sending one
	// here would let the input box accept a submission before the goroutine
	// consuming reqs has even started running.
	//
	// No prompt is passed: the -prompt flag is prefilled into the chat screen's
	// input box for the user to send or edit, and it has nothing to do with
	// getting a model into memory. Threading it through here meant a flag value
	// that failed setup's round-trip check took the whole chat screen down with
	// it, for a string this loop never ran.
	s, err := setup(dir, "")
	if err != nil {
		emit(ctx, out, tui.ChatErr{Err: err})
		return
	}

	cfg := s.gpt.Config
	cache := model.NewKVCache(cfg)
	collector := trace.NewCollector(trace.Header{}, trace.Opts{})
	stop := stopSet(cfg.EOSTokenIDs)

	// seq mirrors what's in the cache, position for position, purely so an
	// attention weight over "position 12" can be turned back into the word
	// that was there. The cache itself only ever holds keys and values.
	seq := make([]int, 0, 256)

	if !emit(ctx, out, tui.ChatStatus("")) {
		return
	}

	var turn uint64
	for {
		var text string
		// Selected on rather than ranged over, so leaving the chat screen ends
		// this goroutine even while it's sitting idle waiting for a prompt that
		// is never going to arrive.
		select {
		case <-ctx.Done():
			return
		case text = <-reqs:
		}

		if text == tui.ClearMarker {
			// /clear already wiped the screen's own transcript; this is what
			// makes that true of the model too — a fresh cache and an empty
			// seq, so the next turn starts the way turn zero did rather than
			// continuing a conversation the screen no longer shows.
			cache = model.NewKVCache(cfg)
			seq = seq[:0]
			turn = 0
			continue
		}

		ids := s.tok.Encode(text)
		if turn > 0 {
			// Turns after the first are a continuation of the transcript, not
			// a new prompt, so they need a separator or the model reads the
			// two as one run-on sentence with no boundary between them.
			ids = append(s.tok.Encode("\n\n"), ids...)
		}
		if len(ids) == 0 {
			if !emit(ctx, out, tui.ChatErr{Err: fmt.Errorf("that produced no tokens to run")}) {
				return
			}
			continue
		}

		logits, err := s.gpt.ForwardCached(ids, cache)
		if err != nil {
			if !emit(ctx, out, tui.ChatErr{Err: err}) {
				return
			}
			continue
		}
		seq = append(seq, ids...)

		sampler := model.NewSampler(model.SampleOpts{Temperature: 0.7, TopK: 20, TopP: 0.95, Seed: turn})
		turn++

		// The prompt plus every reply so far has to fit the context window,
		// same as Generate's own budgeting — this just can't call Generate
		// directly and still get a look at each step's logits and attention.
		budget := chatMaxTokens(s.real)
		if cfg.SequenceLen > 0 {
			budget = min(budget, cfg.SequenceLen-len(seq))
		}

		for n := 0; n < budget; n++ {
			next := sampler.Sample(logits)
			if stop[next] {
				break
			}
			text := s.tok.DecodeSkipSpecial([]int{next})
			if !emit(ctx, out, tui.ChatToken(text)) {
				return
			}
			seq = append(seq, next)

			// Trace only this one decode pass — a single new token attending
			// back over everything cached so far — so the step below describes
			// exactly the token that was just produced, not the whole run. Only
			// done when asked: it's what forces attention heads sequential, so
			// skipping it is what keeps chat at full throughput by default.
			collector.Reset()
			if traceAttention {
				s.gpt.Trace = collector
			}
			logits, err = s.gpt.ForwardCached([]int{next}, cache)
			s.gpt.Trace = nil
			if err != nil {
				break
			}
			if !emit(ctx, out, chatStep(text, collector.Trace(), seq, logits, s.tok)) {
				return
			}
		}
		if err != nil {
			if !emit(ctx, out, tui.ChatErr{Err: err}) {
				return
			}
			continue
		}
		if !emit(ctx, out, tui.ChatDone{}) {
			return
		}
	}
}

// emit hands one message to the chat screen, or gives up if the screen has gone
// away.
//
// out is unbuffered, so a plain send with nobody left reading it blocks this
// goroutine forever — and it would be holding a whole checkpoint resident while
// it did. That is not hypothetical: leaving the chat screen mid-turn stops the
// reader while this loop still has a reply's worth of tokens to hand over.
//
// It only ever bounds the leak to one token, not zero: the forward pass itself
// isn't interruptible, so a cancelled context is noticed at the next token
// boundary rather than during the matmul that's already running.
func emit(ctx context.Context, out chan<- tea.Msg, msg tea.Msg) bool {
	select {
	case out <- msg:
		return true
	case <-ctx.Done():
		return false
	}
}

// chatStep packages one generated token for the inspect tab: what it leaned
// on, drawn from the attention trace of the pass that just fed it in, and what
// the model now ranks as likely to come next.
func chatStep(text string, tr *trace.Trace, seq []int, logits []float32, tok *tokenizer.Tokenizer) tui.ChatStep {
	return tui.ChatStep{
		Token:      text,
		Attention:  attentionSummary(tr, seq, tok),
		Candidates: candidatesOf(model.TopCandidates(logits, 0.7, chatCandidates), tok),
	}
}

// attentionSummary averages every layer and head's attention row into one
// distribution over positions, then names the handful it leaned on most.
//
// Averaging across layers erases real differences between them — an early
// layer often attends locally, a late one to whatever resolved the meaning —
// but the inspect tab has room for one number per token, not one per layer.
// cmd/inspect keeps that detail for anyone who wants to page through it; this
// is the "what mattered, roughly" version next to a live conversation.
func attentionSummary(tr *trace.Trace, seq []int, tok *tokenizer.Tokenizer) []tui.ChatCandidate {
	var sum []float64
	for _, e := range tr.Kind(trace.KindAttention) {
		if len(e.Weights) == 0 {
			continue
		}
		row := e.Weights[0] // Tq is always 1 during decode: one query, one row
		if sum == nil {
			sum = make([]float64, len(row))
		}
		for i, w := range row {
			sum[i] += float64(w)
		}
	}
	if len(sum) == 0 {
		return nil
	}

	type scored struct {
		pos int
		w   float64
	}
	ranked := make([]scored, len(sum))
	var total float64
	for i, w := range sum {
		ranked[i] = scored{i, w}
		total += w
	}
	if total == 0 {
		return nil
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].w > ranked[b].w })
	if len(ranked) > chatCandidates {
		ranked = ranked[:chatCandidates]
	}

	out := make([]tui.ChatCandidate, 0, len(ranked))
	for _, r := range ranked {
		if r.pos >= len(seq) {
			continue // the token attended to itself; there's no earlier word for that position
		}
		out = append(out, tui.ChatCandidate{
			Text: tok.DecodeSkipSpecial([]int{seq[r.pos]}),
			Prob: r.w / total,
		})
	}
	return out
}

func candidatesOf(cs []model.Candidate, tok *tokenizer.Tokenizer) []tui.ChatCandidate {
	out := make([]tui.ChatCandidate, len(cs))
	for i, c := range cs {
		out[i] = tui.ChatCandidate{Text: tok.DecodeSkipSpecial([]int{c.ID}), Prob: c.Prob}
	}
	return out
}

func stopSet(ids []int) map[int]bool {
	out := make(map[int]bool, len(ids))
	for _, id := range ids {
		out[id] = true
	}
	return out
}
