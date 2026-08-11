package history

// SetTestDir points every save/list/count at dir for the lifetime of a test,
// restoring the real ~/.gollama/history location when the test ends. It
// exists so a test that exercises a full chat turn — which saves a
// conversation as a side effect of ChatDone — doesn't write into whichever
// machine happens to be running the suite.
func SetTestDir(t interface{ Cleanup(func()) }, dir string) {
	prev := dirOverride
	dirOverride = dir
	t.Cleanup(func() { dirOverride = prev })
}
