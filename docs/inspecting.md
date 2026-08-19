# Inspecting a run

What each of the inspect screen's four views actually shows, on real Qwen3-0.6B weights given `"The capital of France is"`. The [README](../README.md#inspecting-a-run) covers how to launch it; this is what you'll see once it's open.

The **logit lens** is the reason this exists. It projects the residual stream through the LM head at *every* layer, so you can watch an answer get found:

```text
  layer   prediction          prob     rank      H
  0       " only"            18.1%   #11876   3.43  ██████░░░░░░░░░░░░░░░░░░
  4       " not"             54.6%   #12549   2.37  █████████████░░░░░░░░░░░
  7       " a"               54.8%    #3056   2.15  █████████████░░░░░░░░░░░
  12      " a"               20.7%   #10541   3.56  ████░░░░░░░░░░░░░░░░░░░░
  17      "____"             76.6%    #3597   1.05  ██████████████████░░░░░░
  20      "____"             55.9%      #43   1.26  █████████████░░░░░░░░░░░
  21      "____"             45.1%       #9   1.92  ██████████░░░░░░░░░░░░░░
  22      " Paris"           55.1%       #1   1.87  █████████████░░░░░░░░░░░   ← first leads here
  25      " Paris"           96.4%       #1   0.31  ███████████████████████░
  27      " Paris"           65.7%       #1   2.31  ███████████████░░░░░░░░░
  out     " Paris"           65.7%       #1   2.31  ███████████████░░░░░░░░░
```

Early layers guess generic function words. By layer 7 it knows a noun phrase is coming (`" a"`). Around 17 it's reaching for a blank (`"____"`). **" Paris" only takes the lead at layer 22 of 28**, then sharpens to 96% before the last layer hedges back down.

The two extra columns are there because the top row can't tell you the whole story. `rank` is where the eventual answer stood at that depth: it sits past ten thousand for two thirds of the stack, then goes **#43 → #9 → #1** over three layers. Reading the prediction column alone, " Paris" appears from nowhere at 22; reading the rank, it was climbing hard from 19 onwards, and 22 is where a move already underway crosses the line. `H` is the entropy of the whole distribution in nats, which is the model's confidence *independent of which token is winning* — layer 17 is 76.6% sure of `"____"` at H 1.05, and the last layer drops to H 2.31 while still leading with " Paris". That final hedge is a widening of the whole distribution, not a change of mind.

**Direct logit attribution** answers the question none of the above can: not what the model thought at each depth, but which parts of it are *why*. Every component — each attention head, each MLP, the embedding — adds its own vector into the residual stream, and with the final norm's scaling held at what the finished stream produced, the rest is linear. So an output logit is exactly the sum of the components' individual pushes on it, and each push can be reported separately:

```text
  layer 26   pushing " Paris"

  component     Δlogit   ‖write‖
  head 0        +3.700    43.737              │████████████
  head 1        -1.781    32.796        ██████│
  head 7        -0.629    13.032            ██│
  head 9        -0.144    39.767              │
  …
  mlp           +0.335   114.404              │█

  largest across the whole pass: L26 head 0 +3.70, L24 mlp +3.58, L22 mlp +2.90, L23 mlp +2.46
```

`‖write‖` is how much the component moved the residual stream at all; `Δlogit` is how much of that landed on the answer. They come apart constantly, and that gap is the whole point. Head 9 makes the second-largest write in the layer and contributes **nothing** to " Paris" — whatever it's doing, the answer doesn't depend on it. The MLP writes nearly three times as hard as any head and still only adds +0.34. Head 0, writing less than head 9, supplies +3.70 on its own, and head 1 spends most of a comparable write pushing the answer *down*.

An attention pattern tells you a head looked somewhere. Only this tells you the output depended on it — which is why the two views are worth reading side by side: a striking pattern attached to no push is easy to over-read on its own.

Across the whole pass, three of the five largest contributions are MLPs in layers 22–24 — exactly the stretch where the rank column shows " Paris" climbing from #43 to #1. The heads move the stream harder; the MLPs are where the movement becomes the answer.

The **attention** view shows the sink getting dramatic with depth — at layer 27, token 0 absorbs 94% of every later position's attention:

```text
  layer 27  head 0
                 The _capital      _of  _France      _is
  The          1.000        ·        ·        ·        ·
  _capital     0.983    0.017        ·        ·        ·
  _of          0.962    0.018    0.020        ·        ·
  _France      0.971    0.002    0.009    0.018        ·
  _is          0.862    0.005    0.010    0.027    0.095
  token 0 absorbs 94% of later positions' attention on average
```

Attribution and the attention grid are both still just *watching* the forward pass. **Ablation** is the one view that intervenes: pick a head, force its output to zero before it's merged back into the residual stream, and re-run — if the answer actually moves, the head mattered; if it doesn't, whatever attribution measured wasn't load-bearing. Forcing each of the 448 `(layer, head)` pairs in Qwen3-0.6B to zero, one at a time, on `"The capital of France is"`, exactly one of them flips the top prediction away from `" Paris"` — layer 0, head 3:

```text
  layer   baseline           prob    ablated L0H3       prob
  0       " only"           18.1%    " only"            26.4%
  2       " not"            22.8%    " the"             12.0%
  5       " not"            16.3%    " a"               42.0%
  7       " a"              54.8%    ","                38.1%
  12      " a"              20.7%    ","                28.3%
  17      "____"            76.6%    ","                23.1%
  20      "____"            55.9%    " a"               38.1%
  22      " Paris"          55.1%    " a"               31.7%
  25      " Paris"          96.4%    " a"               19.5%
  27      " Paris"          65.7%    " a"               12.4%
```

The two runs already disagree by layer 2 — long before the baseline commits to " Paris" at layer 22 — so this head is doing something far upstream of where attribution said the decision happens, and the model never recovers " Paris" once it's gone. It's why Ablation leads the welcome menu: the inspect screen's first tab (`1`) shows this comparison live — pick a layer and head with the arrow keys, press `a` to ablate it, and see which layer the two runs first disagree at.

Stepping between tokens is where it gets interesting. On `"The capital of France is"` the answer lands at layer 22. On the *next* token — after "…is Paris." — the model predicts `" The"`, and that one settles by layer 19. Different tokens are decided at different depths.

Temperature is easier to believe when you can watch it work on a fact the model actually knows:

```text
  temperature 0.7        temperature 1.0        temperature 1.5
   93.62%  " Paris"       65.69%  " Paris"       15.37%  " Paris"
    1.02%  " located"      2.78%  " located"      1.87%  " located"
    0.80%  " the"          2.35%  " the"          1.67%  " the"
```
