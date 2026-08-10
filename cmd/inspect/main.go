// Command inspect is an interactive terminal UI for watching a forward pass.
//
// Run it, type a prompt, press enter. It loads the checkpoint once, then traces a
// prefill pass plus one pass per generated token, streaming each into the UI as
// it completes. The logit lens shows what the model would predict at every layer,
// so you can see where in the stack an answer actually gets decided.
//
// With -f it replays a trace recorded earlier by `go run . -trace` instead, which
// needs no checkpoint.
//
// Both paths render the same data, because both go through package trace: the
// live collector and the file writer build identical events. Nothing in model/
// knows this program exists — the dependency arrow only ever points inward.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

func main() {
	var (
		file       = flag.String("f", "", "replay a trace file instead of running the model")
		checkpoint = flag.String("model", "checkpoints/qwen3-0.6b", "checkpoint directory")
		prompt     = flag.String("prompt", "", "pre-fill the prompt field (optional)")
		n          = flag.Int("n", 3, "how many tokens to generate per run")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr,
			"usage: inspect [-model dir] [-prompt text] [-n tokens]\n"+
				"       inspect -f trace.jsonl\n\n"+
				"Type a prompt and press enter to run it.\n"+
				"Keys: 1/2/3 or tab switch view · up/down layer · left/right head\n"+
				"      n/p step between generated tokens · i edit prompt · q quit\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// A bare positional argument is almost certainly a trace file.
	if *file == "" && flag.NArg() == 1 {
		*file = flag.Arg(0)
	}

	app, err := build(*file, *checkpoint, *prompt, *n)
	if err != nil {
		fmt.Fprintf(os.Stderr, "inspect: %v\n", err)
		os.Exit(1)
	}

	if _, err := tea.NewProgram(app, tea.WithAltScreen()).Run(); err != nil {
		fmt.Fprintf(os.Stderr, "inspect: %v\n", err)
		os.Exit(1)
	}
}

func build(file, checkpoint, prompt string, n int) (*app, error) {
	if file != "" {
		tr, err := trace.Open(file)
		if err != nil {
			return nil, err
		}
		if len(tr.Events) == 0 {
			return nil, fmt.Errorf("%s has a header but no events", file)
		}
		return newApp(tr), nil
	}

	if _, err := os.Stat(filepath.Join(checkpoint, "model.safetensors")); err != nil {
		return nil, fmt.Errorf("no checkpoint at %s\n\n"+
			"  download one:  huggingface-cli download Qwen/Qwen3-0.6B --local-dir %s\n"+
			"  or replay a recorded trace:  inspect -f run.jsonl",
			checkpoint, checkpoint)
	}
	if n < 0 {
		return nil, fmt.Errorf("-n is %d, must not be negative", n)
	}

	events := make(chan tea.Msg, 16)
	reqs := make(chan request, 1)
	go runEngine(checkpoint, reqs, events)

	return newLiveApp(events, reqs, prompt, n), nil
}
