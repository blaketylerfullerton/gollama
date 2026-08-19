package main

import (
	"context"
	"fmt"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/blaketylerfullerton/GoLlama/tools/tui"
	"github.com/blaketylerfullerton/GoLlama/tools/watermark"
)

// newWatermarkEngine is the watermark screen's counterpart to newChatEngine.
// It satisfies tui.Engine directly rather than a type of its own — the
// protocol is already generic enough (a prompt in, tea.Msg out) that a
// second declaration would say nothing a comment couldn't.
func newWatermarkEngine() tui.Engine {
	return func(ctx context.Context, dir string, reqs <-chan string, out chan<- tea.Msg) {
		watermarkEngine(ctx, dir, reqs, out)
	}
}

func watermarkEngine(ctx context.Context, dir string, reqs <-chan string, out chan<- tea.Msg) {
	defer close(out)

	// No prompt passed to setup, same reasoning as chatEngine: the -prompt
	// flag is only for prefilling the screen's input box, and shouldn't be
	// able to fail a checkpoint load over a string this loop never runs as
	// typed.
	s, err := setup(dir, "")
	if err != nil {
		emit(ctx, out, tui.WatermarkErr{Err: err})
		return
	}

	cfg := watermarkDemoConfig
	params := fmt.Sprintf("key %#x · %d-token context · %d layers (%d-way)",
		cfg.Key, cfg.ContextSize, cfg.Layers, 1<<cfg.Layers)
	if !emit(ctx, out, tui.WatermarkReady{Params: params}) {
		return
	}

	// turn seeds both generations, same as chatEngine's turn counter seeds
	// its sampler — varied per request so two different prompts (or the same
	// prompt submitted twice) don't draw identical tournaments.
	var turn uint64
	for {
		var prompt string
		select {
		case <-ctx.Done():
			return
		case prompt = <-reqs:
		}

		ids, err := encode(s.tok, prompt)
		if err != nil {
			if !emit(ctx, out, tui.WatermarkErr{Err: err}) {
				return
			}
			continue
		}

		plain, marked, plainScore, markedScore, err := watermarkCompare(
			s, ids, cfg, watermarkMaxTokens, watermarkTemperature, turn)
		turn++
		if err != nil {
			if !emit(ctx, out, tui.WatermarkErr{Err: err}) {
				return
			}
			continue
		}

		result := tui.WatermarkResult{
			Plain:       s.tok.DecodeSkipSpecial(plain),
			Marked:      s.tok.DecodeSkipSpecial(marked),
			PlainScore:  toTUIScore(plainScore),
			MarkedScore: toTUIScore(markedScore),
		}
		if !emit(ctx, out, result) {
			return
		}
	}
}

// toTUIScore copies a watermark.Score into tui.WatermarkScore — package tui
// doesn't import tools/watermark (or engine/model, which that in turn
// depends on), so the fields cross that boundary by value here instead.
func toTUIScore(s watermark.Score) tui.WatermarkScore {
	return tui.WatermarkScore{Positions: s.Positions, MeanG: s.MeanG, Z: s.Z}
}
