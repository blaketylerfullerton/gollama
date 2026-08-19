// Package hf fetches a checkpoint straight off HuggingFace, so that picking an
// uninstalled model in the TUI is the download rather than an instruction to
// go run one yourself in another terminal.
//
// It knows exactly as much about a checkpoint's shape as the loader needs: a
// config.json, a tokenizer.json, an optional generation_config.json, and the
// weights — either one model.safetensors or a model.safetensors.index.json
// plus whichever shards it names. Nothing here parses those files beyond the
// index's weight_map; the loader in engine/model reads them for real once
// they're on disk.
package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// resolveURL is HuggingFace's convention for fetching a file straight out of a
// repo's default branch without cloning it with git-lfs.
const resolveURL = "https://huggingface.co/%s/resolve/main/%s"

// reportEvery throttles progress callbacks to roughly this often, rather than
// once per network read — a 1.5GB file arrives in tens of thousands of 32KB
// chunks, and a channel send for every one of those is wasted work the UI
// couldn't render any faster anyway.
const reportEvery = 250 * time.Millisecond

// Progress is one update on how a Download call is going. File-scoped fields
// describe whatever is currently in flight; Bytes/Total are the running total
// across every file the checkpoint needs.
type Progress struct {
	File      string
	FileIndex int // 1-based
	FileCount int
	FileBytes int64
	FileTotal int64 // -1 when the server didn't say

	Bytes int64
	Total int64 // -1 when at least one file's size is unknown
}

// errNotFound marks a 404, so Download can tell "this file doesn't exist" from
// every other way a fetch can fail.
var errNotFound = errors.New("not found")

// file is one thing Download needs to fetch. optional files that 404 (only
// generation_config.json, some checkpoints don't ship one) are skipped rather
// than failing the whole download.
type file struct {
	name     string
	optional bool
}

// Download fetches everything engine/model and engine/tokenizer need to load
// repo, into dir, reporting progress as report. It returns once every required
// file is on disk, or the first error — including ctx being cancelled, which
// callers use to let a user back out mid-download.
//
// A file already on disk at the size HuggingFace reports is trusted and
// skipped, so a download that was interrupted partway through a multi-shard
// checkpoint resumes at the next shard instead of starting over.
func Download(ctx context.Context, repo, dir string, report func(Progress)) error {
	client := &http.Client{}

	files, err := plan(ctx, client, repo)
	if err != nil {
		return fmt.Errorf("reading %s's file list: %w", repo, err)
	}

	sizes, total := sizesOf(ctx, client, repo, files)

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	var done int64
	for i, f := range files {
		n, err := fetchFile(ctx, client, repo, f.name, dir, sizes[i], func(fb int64) {
			report(Progress{
				File: f.name, FileIndex: i + 1, FileCount: len(files),
				FileBytes: fb, FileTotal: sizes[i],
				Bytes: done + fb, Total: total,
			})
		})
		if err != nil {
			if f.optional && errors.Is(err, errNotFound) {
				continue
			}
			return fmt.Errorf("downloading %s: %w", f.name, err)
		}
		done += n
	}
	return nil
}

// plan decides which files a checkpoint needs: the config and tokenizer every
// Qwen3 checkpoint ships, plus whichever weight files this one uses.
func plan(ctx context.Context, client *http.Client, repo string) ([]file, error) {
	weights, err := weightFiles(ctx, client, repo)
	if err != nil {
		return nil, err
	}
	files := make([]file, 0, len(weights)+3)
	for _, w := range weights {
		files = append(files, file{name: w})
	}
	return append(files,
		file{name: "config.json"},
		file{name: "tokenizer.json"},
		file{name: "generation_config.json", optional: true},
	), nil
}

// shardIndex mirrors the one field of model.safetensors.index.json this
// package cares about: which shard file holds which tensor. Loading the
// tensors themselves is engine/model's job, once they're on disk.
type shardIndex struct {
	WeightMap map[string]string `json:"weight_map"`
}

// weightFiles is model.safetensors for a checkpoint small enough to ship as
// one file, or the index plus every shard it names for one that isn't —
// Qwen3-8B and up split the weights across several model-0000N-of-0000M.safetensors
// files rather than one.
func weightFiles(ctx context.Context, client *http.Client, repo string) ([]string, error) {
	data, status, err := getBytes(ctx, client, repo, "model.safetensors.index.json")
	if err != nil {
		return nil, err
	}
	if status == http.StatusNotFound {
		return []string{"model.safetensors"}, nil
	}
	if status != http.StatusOK {
		return nil, fmt.Errorf("fetching model.safetensors.index.json: %s", http.StatusText(status))
	}

	var idx shardIndex
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("parsing model.safetensors.index.json: %w", err)
	}

	seen := map[string]bool{}
	shards := make([]string, 0, len(idx.WeightMap))
	for _, shard := range idx.WeightMap {
		if !seen[shard] {
			seen[shard] = true
			shards = append(shards, shard)
		}
	}
	sort.Strings(shards)
	return append([]string{"model.safetensors.index.json"}, shards...), nil
}

// sizesOf HEADs every file up front, purely so the progress bar can show a
// real total from the first byte instead of growing the goalpost as each new
// file's size becomes known. A file whose size can't be had — the request
// failed, or the server didn't say — reports -1 and drops out of total rather
// than lying about how much is left.
func sizesOf(ctx context.Context, client *http.Client, repo string, files []file) ([]int64, int64) {
	sizes := make([]int64, len(files))
	var total int64
	known := true
	for i, f := range files {
		n, err := headSize(ctx, client, repo, f.name)
		if err != nil {
			sizes[i] = -1
			known = false
			continue
		}
		sizes[i] = n
		total += n
	}
	if !known {
		total = -1
	}
	return sizes, total
}

func headSize(ctx context.Context, client *http.Client, repo, name string) (int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, fmt.Sprintf(resolveURL, repo, name), nil)
	if err != nil {
		return -1, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return -1, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK || resp.ContentLength < 0 {
		return -1, fmt.Errorf("no size for %s", name)
	}
	return resp.ContentLength, nil
}

// getBytes fetches name whole, for files small enough to read into memory
// before deciding what they mean — right now just the shard index. It reports
// the status code rather than turning a 404 into an error, since the one
// caller needs to tell "this repo has no index" from every other failure.
func getBytes(ctx context.Context, client *http.Client, repo, name string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf(resolveURL, repo, name), nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode, nil
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, err
	}
	return data, resp.StatusCode, nil
}

// fetchFile downloads one file into dir, as dir/name.part until it's whole and
// then renamed into place — so a download killed partway through never leaves
// something that looks finished. size is what headSize found, used only to
// skip a file that's already there at exactly that length.
func fetchFile(ctx context.Context, client *http.Client, repo, name, dir string, size int64, onByte func(int64)) (int64, error) {
	dest := filepath.Join(dir, name)
	if size > 0 {
		if info, err := os.Stat(dest); err == nil && info.Size() == size {
			onByte(size)
			return 0, nil
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf(resolveURL, repo, name), nil)
	if err != nil {
		return 0, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusNotFound {
		return 0, errNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("%s", http.StatusText(resp.StatusCode))
	}

	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return 0, err
	}

	n, copyErr := io.Copy(f, &countingReader{r: resp.Body, onByte: onByte})
	closeErr := f.Close()
	switch {
	case copyErr != nil:
		_ = os.Remove(tmp)
		return n, copyErr
	case closeErr != nil:
		_ = os.Remove(tmp)
		return n, closeErr
	}
	if err := os.Rename(tmp, dest); err != nil {
		return n, err
	}
	onByte(n)
	return n, nil
}

// countingReader calls onByte with the running total as it's read through,
// throttled to reportEvery so a fast local connection doesn't turn every 32KB
// io.Copy chunk into a channel send.
type countingReader struct {
	r      io.Reader
	n      int64
	onByte func(int64)
	lastAt time.Time
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	if now := time.Now(); now.Sub(c.lastAt) >= reportEvery {
		c.lastAt = now
		c.onByte(c.n)
	}
	return n, err
}
