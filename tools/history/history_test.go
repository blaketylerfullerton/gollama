package history

import (
	"testing"
	"time"
)

func TestSaveAndList(t *testing.T) {
	SetTestDir(t, t.TempDir())

	c := Conversation{
		ID:        NewID(time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)),
		Label:     "Qwen3-0.6B",
		StartedAt: time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC),
		Turns:     []Entry{{You: "hi", Model: "hello"}},
	}
	if err := Save(c); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("List returned %d conversations, want 1", len(got))
	}
	if got[0].Label != c.Label || len(got[0].Turns) != 1 || got[0].Turns[0].You != "hi" {
		t.Errorf("List() = %+v, want a round trip of %+v", got[0], c)
	}
	if n := Count(); n != 1 {
		t.Errorf("Count() = %d, want 1", n)
	}
}

// Saving twice under the same ID overwrites rather than appending a second
// file — the chat screen calls Save once per completed turn, and a growing
// pile of near-duplicate files for one conversation would be a bug, not a
// feature.
func TestSaveOverwrites(t *testing.T) {
	SetTestDir(t, t.TempDir())

	id := NewID(time.Now())
	Save(Conversation{ID: id, Turns: []Entry{{You: "one"}}})
	Save(Conversation{ID: id, Turns: []Entry{{You: "one"}, {You: "two"}}})

	got, err := List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("List returned %d conversations, want 1", len(got))
	}
	if len(got[0].Turns) != 2 {
		t.Errorf("Turns = %d, want 2 (the second save should replace the first)", len(got[0].Turns))
	}
}

// A machine that has never saved a conversation shouldn't error just because
// the directory doesn't exist yet.
func TestListEmpty(t *testing.T) {
	SetTestDir(t, t.TempDir()+"/does-not-exist")
	got, err := List()
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("List() = %v, want empty", got)
	}
	if n := Count(); n != 0 {
		t.Errorf("Count() = %d, want 0", n)
	}
}
