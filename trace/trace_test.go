package trace

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/blaketylerfullerton/GoLlama/model"
)

func testHeader() Header {
	return Header{
		Model:  "test",
		Prompt: "hi there",
		Tokens: []Token{{ID: 1, Text: "hi"}, {ID: 2, Text: " there"}},
		Config: ModelInfo{NLayer: 2, NEmbed: 4, NHead: 2, NKVHead: 1, HeadDim: 2, VocabSize: 8},
	}
}

// The Writer must satisfy both interfaces, or the engine silently won't ask it
// for intermediate predictions.
var (
	_ model.Tracer          = (*Writer)(nil)
	_ model.LogitLensTracer = (*Writer)(nil)
)

func TestRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w, err := NewWriter(&buf, testHeader(), Opts{Vocab: func(id int) string { return "tok" }})
	if err != nil {
		t.Fatal(err)
	}

	w.Stage(-1, "token embeddings", [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}})
	w.Stage(0, "input norm", [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}})
	w.Attention(0, 1, [][]float32{{1}, {0.4, 0.6}})
	w.Rotary(0, 1, []float32{1, 0}, []float32{0, 1})
	w.Note(0, "gate %d%% negative", 51)
	w.LogitLens(0, []float32{0, 5, 0, 0, 0, 0, 0, 0})

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if w.Events() != 6 {
		t.Errorf("recorded %d events, want 6", w.Events())
	}

	tr, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if tr.Header.Version != FormatVersion {
		t.Errorf("version is %d, want %d", tr.Header.Version, FormatVersion)
	}
	if tr.Header.Prompt != "hi there" || len(tr.Header.Tokens) != 2 {
		t.Errorf("header did not survive: %+v", tr.Header)
	}
	if len(tr.Events) != 6 {
		t.Fatalf("read %d events, want 6", len(tr.Events))
	}

	// Sequence numbers must be dense and ordered, so a reader can rely on them.
	for i, e := range tr.Events {
		if e.Seq != i {
			t.Errorf("event %d has seq %d", i, e.Seq)
		}
	}

	att, ok := tr.LayerEvent(0, KindAttention, 1)
	if !ok {
		t.Fatal("attention event missing")
	}
	if len(att.Weights) != 2 || att.Weights[1][1] != 0.6 {
		t.Errorf("attention weights did not survive: %v", att.Weights)
	}

	lens := tr.Kind(KindLogitLens)
	if len(lens) != 1 || len(lens[0].Top) == 0 {
		t.Fatalf("logit lens missing: %+v", lens)
	}
	// Index 1 has the only nonzero logit, so it must win.
	if lens[0].Top[0].ID != 1 {
		t.Errorf("top candidate is %d, want 1", lens[0].Top[0].ID)
	}
	if lens[0].Top[0].Text != "tok" {
		t.Errorf("Vocab was not applied, got %q", lens[0].Top[0].Text)
	}
}

// The Tracer contract says implementations must not retain slices, because the
// engine may reuse the buffer. The Writer therefore has to copy — this checks it
// does, by scribbling over the caller's slices immediately afterwards.
func TestWriterCopiesBeforeReturning(t *testing.T) {
	var buf bytes.Buffer
	w, err := NewWriter(&buf, testHeader(), Opts{})
	if err != nil {
		t.Fatal(err)
	}

	stage := [][]float32{{1, 2, 3, 4}}
	weights := [][]float32{{1}, {0.25, 0.75}}
	before, after := []float32{3, 4}, []float32{4, 3}

	w.Stage(0, "s", stage)
	w.Attention(0, 0, weights)
	w.Rotary(0, 0, before, after)

	// Simulate the engine reusing every buffer it just handed over.
	for i := range stage[0] {
		stage[0][i] = -999
	}
	for i := range weights {
		for j := range weights[i] {
			weights[i][j] = -999
		}
	}
	for i := range before {
		before[i], after[i] = -999, -999
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	tr, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range tr.Events {
		for _, v := range e.Preview {
			if v == -999 {
				t.Fatal("Stage preview aliased the caller's slice")
			}
		}
		for _, row := range e.Weights {
			for _, v := range row {
				if v == -999 {
					t.Fatal("Attention weights aliased the caller's slice")
				}
			}
		}
		for _, v := range append(append([]float32{}, e.Before...), e.After...) {
			if v == -999 {
				t.Fatal("Rotary vectors aliased the caller's slice")
			}
		}
	}
}

func TestPreviewIsClipped(t *testing.T) {
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, testHeader(), Opts{PreviewDims: 3})

	row := make([]float32, 100)
	for i := range row {
		row[i] = float32(i)
	}
	w.Stage(0, "wide", [][]float32{row})
	w.Close()

	tr, err := Read(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	e := tr.Events[0]
	if len(e.Preview) != 3 {
		t.Errorf("preview has %d values, want 3", len(e.Preview))
	}
	// Dims must still report the true width, not the clipped one.
	if e.Dims != 100 {
		t.Errorf("Dims is %d, want 100", e.Dims)
	}
}

// Attention weights are O(T²) per head per layer, so long sequences are skipped
// rather than allowed to dominate the file.
func TestAttentionSkippedForLongSequences(t *testing.T) {
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, testHeader(), Opts{MaxAttentionTokens: 3})

	short := [][]float32{{1}, {0.5, 0.5}}
	long := make([][]float32, 10)
	for i := range long {
		long[i] = make([]float32, i+1)
	}
	w.Attention(0, 0, short)
	w.Attention(1, 0, long)
	w.Close()

	tr, _ := Read(bytes.NewReader(buf.Bytes()))
	if got := len(tr.Kind(KindAttention)); got != 1 {
		t.Errorf("recorded %d attention events, want 1 (the long one should be skipped)", got)
	}
}

// A stage with no rows would index out of bounds on x[0].
func TestStageIgnoresEmptyInput(t *testing.T) {
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, testHeader(), Opts{})
	w.Stage(0, "empty", nil)
	w.Stage(0, "empty", [][]float32{})
	w.Close()

	tr, _ := Read(bytes.NewReader(buf.Bytes()))
	if len(tr.Events) != 0 {
		t.Errorf("recorded %d events for empty input, want 0", len(tr.Events))
	}
}

func TestByLayerSeparatesOutsideStack(t *testing.T) {
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, testHeader(), Opts{})
	w.Stage(-1, "token embeddings", [][]float32{{1}})
	w.Stage(0, "input norm", [][]float32{{1}})
	w.Stage(1, "input norm", [][]float32{{1}})
	w.Stage(-1, "logits", [][]float32{{1}})
	w.Close()

	tr, _ := Read(bytes.NewReader(buf.Bytes()))
	layers, outside := tr.ByLayer()
	if len(outside) != 2 {
		t.Errorf("got %d events outside the stack, want 2", len(outside))
	}
	if len(layers) != 2 || len(layers[0]) != 1 || len(layers[1]) != 1 {
		t.Errorf("layer grouping is wrong: %v", layers)
	}
}

// --- reader robustness ------------------------------------------------------

func TestReadRejectsBadInput(t *testing.T) {
	for name, body := range map[string]string{
		"empty":         "",
		"bad header":    "not json\n",
		"bad event":     `{"version":1}` + "\n" + "not json\n",
		"newer version": `{"version":9999}` + "\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := Read(strings.NewReader(body)); err == nil {
				t.Error("expected an error")
			}
		})
	}
}

func TestReadSkipsBlankLines(t *testing.T) {
	body := `{"version":1,"model":"m"}` + "\n\n" + `{"seq":0,"kind":"note","layer":0,"text":"x"}` + "\n\n"
	tr, err := Read(strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.Events) != 1 {
		t.Errorf("got %d events, want 1", len(tr.Events))
	}
}

func TestOpenMissingFile(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "nope.jsonl")); err == nil {
		t.Error("expected an error for a missing file")
	}
}

func TestLayerEventMissReturnsFalse(t *testing.T) {
	var buf bytes.Buffer
	w, _ := NewWriter(&buf, testHeader(), Opts{})
	w.Attention(0, 0, [][]float32{{1}})
	w.Close()
	tr, _ := Read(bytes.NewReader(buf.Bytes()))

	if _, ok := tr.LayerEvent(0, KindAttention, 7); ok {
		t.Error("expected no match for a head that was never recorded")
	}
	if _, ok := tr.LayerEvent(5, KindAttention, 0); ok {
		t.Error("expected no match for a layer that was never recorded")
	}
}
