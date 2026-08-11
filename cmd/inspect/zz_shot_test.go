package main

import (
	"os"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/termenv"

	"github.com/blaketylerfullerton/GoLlama/tools/trace"
)

func TestZZShot(t *testing.T) {
	out := os.Getenv("SHOT_OUT")
	if out == "" {
		t.Skip("no SHOT_OUT")
	}
	lipgloss.SetColorProfile(termenv.TrueColor)
	tr, err := trace.Open("../../run.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	a := newApp(tr)
	a.w, a.h = 110, 34
	var buf []byte
	for _, v := range []view{viewLens, viewAttention, viewAttribution, viewStages} {
		a.view = v
		a.layer = 14
		buf = append(buf, a.View()...)
		buf = append(buf, "\n\n"...)
	}
	if err := os.WriteFile(out, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}
