package model

import (
	"go/parser"
	"go/token"
	"io/fs"
	"strconv"
	"strings"
	"testing"
)

const modulePath = "github.com/blaketylerfullerton/GoLlama"

// imports lists the non-test imports of every .go file in a directory.
func imports(t *testing.T, dir string) map[string][]string {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parsing %s: %v", dir, err)
	}

	out := make(map[string][]string)
	for _, pkg := range pkgs {
		for name, file := range pkg.Files {
			for _, spec := range file.Imports {
				path, err := strconv.Unquote(spec.Path.Value)
				if err != nil {
					t.Fatalf("%s: bad import %s", name, spec.Path.Value)
				}
				out[name] = append(out[name], path)
			}
		}
	}
	return out
}

// isStdlib is true for standard library paths. Every module path has a dot in
// its first element (a hostname); no stdlib path does.
func isStdlib(path string) bool {
	first, _, _ := strings.Cut(path, "/")
	return !strings.Contains(first, ".")
}

// The engine must stay dependency-free. The terminal UI pulls in bubbletea and
// lipgloss, and the whole point of keeping it under tools/ and cmd/ is that
// those never reach the packages doing inference.
func TestEngineHasNoThirdPartyDependencies(t *testing.T) {
	for _, dir := range []string{".", "../tokenizer"} {
		for file, paths := range imports(t, dir) {
			for _, path := range paths {
				if isStdlib(path) || strings.HasPrefix(path, modulePath) {
					continue
				}
				t.Errorf("%s imports third-party package %q — the engine must stay stdlib-only",
					file, path)
			}
		}
	}
}

// The trace format is stdlib-only too. It isn't part of the engine — it's how
// the engine's internals get out to the TUI — but a trace file should be
// readable by anything, without dragging a rendering library in behind it.
func TestTraceHasNoThirdPartyDependencies(t *testing.T) {
	for file, paths := range imports(t, "../../tools/trace") {
		for _, path := range paths {
			if isStdlib(path) || strings.HasPrefix(path, modulePath) {
				continue
			}
			t.Errorf("%s imports third-party package %q — the trace format must stay stdlib-only",
				file, path)
		}
	}
}

// The dependency arrow points one way: tools/ may depend on engine/, never the
// reverse. That split is the whole reason for the two top-level folders — a
// reader can open engine/ knowing nothing in it reaches back out to a terminal.
// Breaking this is what turns an observability layer into a thing you can't
// refactor around.
func TestEngineDoesNotImportTooling(t *testing.T) {
	forbidden := []string{modulePath + "/tools", modulePath + "/cmd"}

	for _, dir := range []string{".", "../tokenizer"} {
		for file, paths := range imports(t, dir) {
			for _, path := range paths {
				for _, bad := range forbidden {
					if strings.HasPrefix(path, bad) {
						t.Errorf("%s imports %q — engine/ must not depend on anything that consumes it",
							file, path)
					}
				}
			}
		}
	}
}

// The tokenizer stands alone: it converts text to ids and back, and knows
// nothing about running a model.
func TestTokenizerDoesNotImportModel(t *testing.T) {
	for file, paths := range imports(t, "../tokenizer") {
		for _, path := range paths {
			if strings.HasPrefix(path, modulePath) {
				t.Errorf("%s imports %q — the tokenizer should be self-contained", file, path)
			}
		}
	}
}
