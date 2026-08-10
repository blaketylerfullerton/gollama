package main

import (
	"fmt"
	"sort"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/engine/tokenizer"
	"github.com/blaketylerfullerton/GoLlama/tools/trace"
	"github.com/blaketylerfullerton/GoLlama/tools/tui"
)

// runChatUI is the third screen: after the splash and the picker, a
// conversation with whatever was chosen.
//
// It owns the one thing tui.Chat deliberately doesn't: what "generate" means.
// The screen just streams strings and ranked lists; this function is where
// those come from a real *model.GPT (or the random one) instead.
//
// label and arch come from the picker's own catalog entry rather than from
// loading anything — chosen.Name and chosen.Arch are already known the moment
// enter is pressed there. That's what lets this open the chat screen's alt
// screen immediately: loading the checkpoint (a real multi-second read for the
// 0.6B) happens inside chatEngine, after the screen is already up and showing
// "loading…", instead of blocking in plain terminal between the picker closing
// and the chat screen opening. Doing it the other way around is what used to
// make picking a model look like the whole program had quit and relaunched.
func runChatUI(dir, label, prompt string, arch tui.Arch) error {
	events := make(chan tea.Msg)
	reqs := make(chan string, 1)
	go chatEngine(dir, prompt, reqs, events)

	chat := tui.NewChat(label, arch, events, reqs, prompt)
	_, err := tea.NewProgram(chat, tea.WithAltScreen()).Run()
	return err
}

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

// chatEngine loads the checkpoint, then turns typed lines into generated
// ones. It runs for the lifetime of the chat screen, holding one KV cache
// across every turn: what you type second is a continuation of what the model
// already said after the first, not a fresh prompt, so the cache has to carry
// the whole conversation rather than restart it.
//
// It also traces its own decode step, one token at a time, purely so the
// inspect tab has something real to show: what a token attended to, and what
// the model ranked as likely to follow it. That's the same instrumentation
// cmd/inspect runs, just read out as two short lists instead of full matrices.
func chatEngine(dir, prompt string, reqs <-chan string, out chan<- tea.Msg) {
	defer close(out)

	// No status is sent before the checkpoint is loaded: tui.Chat already opens
	// in its loading phase showing "loading…", and any ChatStatus — even one
	// that just repeats that same text — flips it straight to idle. Sending one
	// here would let the input box accept a submission before the goroutine
	// consuming reqs has even started running.
	s, err := setup(dir, prompt)
	if err != nil {
		out <- tui.ChatErr{Err: err}
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

	out <- tui.ChatStatus("")

	var turn uint64
	for text := range reqs {
		ids := s.tok.Encode(text)
		if turn > 0 {
			// Turns after the first are a continuation of the transcript, not
			// a new prompt, so they need a separator or the model reads the
			// two as one run-on sentence with no boundary between them.
			ids = append(s.tok.Encode("\n\n"), ids...)
		}
		if len(ids) == 0 {
			out <- tui.ChatErr{Err: fmt.Errorf("that produced no tokens to run")}
			continue
		}

		logits, err := s.gpt.ForwardCached(ids, cache)
		if err != nil {
			out <- tui.ChatErr{Err: err}
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
			out <- tui.ChatToken(text)
			seq = append(seq, next)

			// Trace only this one decode pass — a single new token attending
			// back over everything cached so far — so the step below describes
			// exactly the token that was just produced, not the whole run.
			collector.Reset()
			s.gpt.Trace = collector
			logits, err = s.gpt.ForwardCached([]int{next}, cache)
			s.gpt.Trace = nil
			if err != nil {
				break
			}
			out <- chatStep(text, collector.Trace(), seq, logits, s.tok)
		}
		if err != nil {
			out <- tui.ChatErr{Err: err}
			continue
		}
		out <- tui.ChatDone{}
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
