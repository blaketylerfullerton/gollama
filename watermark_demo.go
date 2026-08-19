package main

import (
	"fmt"

	"github.com/blaketylerfullerton/GoLlama/engine/model"
	"github.com/blaketylerfullerton/GoLlama/tools/watermark"
)

// watermarkDemoConfig is fixed rather than exposed as flags: the point of
// this demo is showing the mechanism, not tuning it, and a stable
// key/context/depth keeps the printed comparison from moving underfoot. Both
// the -watermark CLI flag and the Watermark screen share it, via
// watermarkCompare.
var watermarkDemoConfig = watermark.Config{
	Key:         0xC0FFEE_D15C0_5EED, // not a real secret — this is a demo
	ContextSize: 4,
	Layers:      4, // 16-way tournament per token
}

// Tournament sampling can only bias a draw where the model itself is
// genuinely undecided — at low temperature a confident model leaves the
// tournament nothing to work with, and the detector's z-score barely clears
// 1. 1.0 keeps enough entropy in play for the signal to show up clearly (z
// well past the ~4 line) without the output degrading into noise, which is
// what happens if temperature climbs much higher than this in search of an
// even stronger signal.
const (
	watermarkMaxTokens   = 80
	watermarkTemperature = 1.0
)

// watermarkCompare runs ids through ordinary sampling and through
// watermark.Generate under the same temperature and seed, then scores both
// with the same Config's detector. Shared by runWatermarkDemo and
// watermarkEngine so the CLI flag and the TUI screen can't quietly drift
// into showing two different comparisons.
func watermarkCompare(s *session, ids []int, cfg watermark.Config, maxTokens int, temperature float64, seed uint64) (
	plain, marked []int, plainScore, markedScore watermark.Score, err error) {

	plain, err = s.gpt.Generate(ids, model.GenerateOpts{
		MaxTokens:  maxTokens,
		SampleOpts: model.SampleOpts{Temperature: temperature, TopK: 20, TopP: 0.95, Seed: seed},
	})
	if err != nil {
		return
	}
	marked, err = watermark.Generate(s.gpt, cfg, ids, watermark.GenerateOpts{
		Temperature: temperature,
		MaxTokens:   maxTokens,
		Seed:        seed,
	})
	if err != nil {
		return
	}
	plainScore = watermark.Detect(cfg, append(append([]int{}, ids...), plain...))
	markedScore = watermark.Detect(cfg, append(append([]int{}, ids...), marked...))
	return
}

// runWatermarkDemo generates prompt twice — once with the engine's ordinary
// sampler, once with watermark.Generate's tournament sampling — then runs
// the detector on both. The point it's making: both reads are fluent model
// output, but only one of them carries a statistical signature the detector
// can pick out without ever seeing how it was made.
func runWatermarkDemo(prompt string) error {
	s, err := setup(checkpointDir, prompt)
	if err != nil {
		return err
	}
	cfg := watermarkDemoConfig

	fmt.Printf("prompt  %q\n", prompt)
	fmt.Printf("SynthID-Text demo — key %#x, %d-token context, %d tournament layers (%d-way)\n\n",
		cfg.Key, cfg.ContextSize, cfg.Layers, 1<<cfg.Layers)

	plain, marked, plainScore, markedScore, err := watermarkCompare(
		s, s.ids, cfg, watermarkMaxTokens, watermarkTemperature, 1)
	if err != nil {
		return err
	}

	fmt.Println(rule("plain (ordinary sampling)"))
	fmt.Println(s.tok.DecodeSkipSpecial(plain))

	fmt.Println("\n" + rule("watermarked (tournament sampling)"))
	fmt.Println(s.tok.DecodeSkipSpecial(marked))

	fmt.Println("\n" + rule("detector"))
	fmt.Printf("  plain        mean g %.3f   z %6.2f   (%d scored positions)\n",
		plainScore.MeanG, plainScore.Z, plainScore.Positions)
	fmt.Printf("  watermarked  mean g %.3f   z %6.2f   (%d scored positions)\n",
		markedScore.MeanG, markedScore.Z, markedScore.Positions)
	fmt.Println("\n  z above ~4 is the usual line for \"almost certainly watermarked\" — " +
		"ordinary text has no reason to land there.\n" +
		"  (this only works because the model was genuinely uncertain at each step —" +
		" a greedy, low-temperature generation leaves the tournament nothing to bias,\n" +
		"  and the signal weakens accordingly; that entropy-dependence is a real property" +
		" of tournament sampling, not just this demo's.)")
	return nil
}
