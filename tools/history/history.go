// Package history persists chat transcripts to disk so a past conversation
// can be reopened after the program that produced it has already quit.
//
// Storage is plain JSON, one file per conversation, under
// ~/.gollama/history — legible without this package, and easy to prune by
// hand if it grows past what's worth keeping.
package history

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// Entry is one exchange: what was typed, and however much of the model's
// reply had landed by the time the conversation was saved.
type Entry struct {
	You   string `json:"you"`
	Model string `json:"model"`
}

// Conversation is everything needed to show a past session again: who it was
// with and when, plus every turn.
type Conversation struct {
	ID        string    `json:"id"`
	Label     string    `json:"label"` // the model name shown in the chat header
	StartedAt time.Time `json:"started_at"`
	Turns     []Entry   `json:"turns"`
}

// NewID names a conversation from when it started. Seconds are enough
// resolution — two conversations starting in the same second would need to
// come from two processes racing each other, which this program never does.
func NewID(t time.Time) string { return t.Format("20060102-150405") }

// dirOverride, when set, replaces the default ~/.gollama/history location.
// Tests use this so exercising a save doesn't write into whatever machine
// happens to be running them; nothing in the running program ever sets it.
var dirOverride string

// dir is where conversations live. It isn't created here — callers that only
// read (List, Count) shouldn't conjure an empty directory into existing on a
// machine that has never saved anything.
func dir() (string, error) {
	if dirOverride != "" {
		return dirOverride, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".gollama", "history"), nil
}

// Save writes c to disk, creating the history directory if this is the first
// conversation saved. Called again with the same ID, it overwrites — a chat
// screen calls this once per completed turn so a crash mid-conversation loses
// at most the turn in flight, not everything before it.
func Save(c Conversation) error {
	d, err := dir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(d, 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(d, c.ID+".json"), data, 0o644)
}

// List returns every saved conversation, newest first. A history directory
// that doesn't exist yet is an empty list, not an error — that's the normal
// state before the first conversation is ever saved.
func List() ([]Conversation, error) {
	d, err := dir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(d)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []Conversation
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(d, e.Name()))
		if err != nil {
			continue // a file another process is still writing isn't worth failing the whole list over
		}
		var c Conversation
		if err := json.Unmarshal(data, &c); err != nil {
			continue
		}
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].StartedAt.After(out[j].StartedAt) })
	return out, nil
}

// Count is List without paying to parse every file — just how many there are,
// for a menu blurb that only needs the number.
func Count() int {
	d, err := dir()
	if err != nil {
		return 0
	}
	entries, err := os.ReadDir(d)
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			n++
		}
	}
	return n
}
