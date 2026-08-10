package main

import (
	"fmt"
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/tools/tui"
)

// runChatUI is the third screen: after the splash and the picker, a
// conversation with whatever was chosen.
//
// It owns the one thing tui.Chat deliberately doesn't: what "generate" means.
// The screen just streams strings: this function is where those strings come
// from a real *model.GPT (or the random one) instead.
func runChatUI(s *session) error {
	events := make(chan tea.Msg)
	reqs := make(chan string, 1)
	go chatEngine(s, reqs, events)

	label := filepath.Base(s.dir)
	if !s.real {
		label = "the tiny random model — no checkpoint was loaded"
	}
	chat := tui.NewChat(label, events, reqs, s.prompt)
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

// chatEngine turns typed lines into generated ones. It runs for the lifetime
// of the chat screen, holding one KV cache across every turn: what you type
// second is a continuation of what the model already said after the first,
// not a fresh prompt, so the cache has to carry the whole conversation rather
// than restart it.
func chatEngine(s *session, reqs <-chan string, out chan<- tea.Msg) {
	defer close(out)

	cache := model.NewKVCache(s.gpt.Config)
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

		_, err := s.gpt.Generate(ids, model.GenerateOpts{
			MaxTokens:  chatMaxTokens(s.real),
			SampleOpts: model.SampleOpts{Temperature: 0.7, TopK: 20, TopP: 0.95, Seed: turn},
			Cache:      cache,
			OnToken:    func(id int) { out <- tui.ChatToken(s.tok.DecodeSkipSpecial([]int{id})) },
		})
		turn++
		if err != nil {
			out <- tui.ChatErr{Err: err}
			continue
		}
		out <- tui.ChatDone{}
	}
}
