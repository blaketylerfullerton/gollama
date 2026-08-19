# Tests

What `go test ./...` actually covers, beyond the headline count. See the [README](../README.md#tests) for how to run it.

The unit tests build synthetic checkpoints on disk using real HuggingFace tensor names, over a tiny config that deliberately keeps `HeadDim != NEmbed/NHead` and `NKVHead < NHead` — so anything that wrongly derives head dim, or confuses query width with kv width, fails there rather than in a 600MB file.

A few worth naming:

- `TestCachedMatchesUncachedIncrementally` — feeds tokens one at a time and compares against a full uncached pass over the same prefix at every step. Position drift compounds, so a late-position offset bug surfaces here even when positions 0 and 1 happen to agree. `TestRealCheckpointCachedMatchesUncached` does the same across 28 layers and the real rotary tables.
- `TestForwardIsCausal` — truncating the input must not change the logits for the positions that remain, because no position may attend to the future. The strongest correctness property available without a reference implementation.
- `TestLoaderRejectsWrongHeadDim` — feeds the loader the value `NEmbed/NHead` would have produced and confirms it fails loudly naming `q_proj`.
- The sampling tests check the *distribution*, not just the range: 20000 draws against known probabilities.
- `TestGenerateDoesNotMutatePrompt` — the naive implementation passes this only when the caller's slice has no spare capacity.
- The tokenizer tests run against the real `tokenizer.json` as well as fixtures, to catch format drift in a file I don't control.
- `TestWriterCopiesBeforeReturning` enforces the Tracer contract by scribbling `-999` over every slice right after handing it to the trace writer, then checking none of it reached the file. Without this, buffer reuse in the engine would silently corrupt traces.
- `TestEngineHasNoThirdPartyDependencies` and `TestModelDoesNotImportPresentation` parse the import graph and fail if the engine ever grows a dependency or starts depending on something that consumes it.
- The inspector's views are rendered at four terminal sizes, including 20x5, because off-by-one errors in layout code are invisible until someone resizes.
