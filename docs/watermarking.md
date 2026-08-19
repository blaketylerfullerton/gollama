# Watermarking

How `tools/watermark/`'s tournament sampling and detector actually work. The [README](../README.md#watermarking) covers the demo command and its output; this is the mechanism behind it.

Ordinary sampling draws one token from the distribution. Tournament sampling draws `2^Layers` i.i.d. candidates instead, then runs a single-elimination bracket: round `l` compares two candidates by `gValue(seed, l, token)`, a pseudorandom score in `[0,1)` seeded from a secret `Key` and the last `ContextSize` tokens, and the higher score wins. The survivor is still a legitimate draw from the model's own distribution — nothing is truncated or reweighted — but it's biased toward high-`g` tokens, and that bias is exactly what `Detect` looks for: recompute each emitted token's `g`-value against the same `Key`, average over every scored position and tournament layer, and compare to the 0.5 an unwatermarked text would sit at. The gap becomes a z-score, since under the null each `g`-value behaves as an independent `Uniform[0,1)` draw.

The signal depends on the model actually being uncertain at each step: greedy or low-temperature decoding leaves the tournament nothing to bias, since there's no entropy in the draw for the bracket to skew. That's why the demo runs at `Temperature: 1.0` rather than the lower values elsewhere in this repo — a real property of tournament sampling, not an artifact of the demo.

There's no `TopK`/`TopP` in `watermark.GenerateOpts` — either would truncate the candidate pool before the watermark gets a say, which is exactly the distortion tournament sampling exists to avoid.
