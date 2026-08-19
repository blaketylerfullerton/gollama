package trace

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Trace is a whole recorded run, loaded into memory.
type Trace struct {
	Header Header
	Events []Event
}

// Open reads a trace file.
func Open(path string) (*Trace, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening trace: %w", err)
	}
	defer func() { _ = f.Close() }()
	return Read(f)
}

// Read parses a trace from any reader: header line first, then one event per
// line.
func Read(r io.Reader) (*Trace, error) {
	sc := bufio.NewScanner(r)
	// Attention weight rows can be long, so the default 64KB line cap isn't
	// enough for a wide prompt.
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)

	if !sc.Scan() {
		if err := sc.Err(); err != nil {
			return nil, fmt.Errorf("reading trace header: %w", err)
		}
		return nil, fmt.Errorf("trace is empty")
	}

	var t Trace
	if err := json.Unmarshal(sc.Bytes(), &t.Header); err != nil {
		return nil, fmt.Errorf("parsing trace header: %w", err)
	}
	if t.Header.Version > FormatVersion {
		return nil, fmt.Errorf("trace is format version %d, this build understands up to %d",
			t.Header.Version, FormatVersion)
	}

	for line := 2; sc.Scan(); line++ {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var e Event
		if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
			return nil, fmt.Errorf("parsing trace line %d: %w", line, err)
		}
		t.Events = append(t.Events, e)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("reading trace: %w", err)
	}
	return &t, nil
}

// ByLayer groups events by layer index. Layer -1 (embeddings, final norm,
// logits) is returned separately since it isn't part of the stack.
func (t *Trace) ByLayer() (layers map[int][]Event, outside []Event) {
	layers = make(map[int][]Event)
	for _, e := range t.Events {
		if e.Layer < 0 {
			outside = append(outside, e)
			continue
		}
		layers[e.Layer] = append(layers[e.Layer], e)
	}
	return layers, outside
}

// Kind returns every event of one kind, in order.
func (t *Trace) Kind(k Kind) []Event {
	var out []Event
	for _, e := range t.Events {
		if e.Kind == k {
			out = append(out, e)
		}
	}
	return out
}

// Attributions returns the attribution events for one layer, in the order the
// components ran: every attention head, then the MLP. Layer -1 holds the token
// embedding's own contribution.
func (t *Trace) Attributions(layer int) []Event {
	var out []Event
	for _, e := range t.Events {
		if e.Kind == KindAttribution && e.Layer == layer {
			out = append(out, e)
		}
	}
	return out
}

// EffectOn returns this event's effect on one token, and whether the token was
// among the ones attributed. Attribution covers only the candidates that were in
// contention, so a miss is normal rather than an error.
func (e Event) EffectOn(id int) (float32, bool) {
	for _, f := range e.Effects {
		if f.ID == id {
			return f.Logit, true
		}
	}
	return 0, false
}

// LayerEvent finds a single event by layer, kind and head. head is ignored for
// kinds that don't have one. Returns false when there's no match.
func (t *Trace) LayerEvent(layer int, k Kind, head int) (Event, bool) {
	for _, e := range t.Events {
		if e.Layer != layer || e.Kind != k {
			continue
		}
		if (k == KindAttention || k == KindRotary) && e.Head != head {
			continue
		}
		return e, true
	}
	return Event{}, false
}
