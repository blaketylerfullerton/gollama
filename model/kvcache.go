package model

import "fmt"

// KVCache stores the key and value vectors for every position the model has
// already processed, so generating token N+1 doesn't recompute the first N.
//
// It deliberately isn't a field on GPT. GPT is just weights; the cache is
// per-conversation state. Keeping them apart means one loaded model can serve
// many independent generations at once.
//
// Only k and v are cached, never q. A query is used once, by the token that
// issued it, and then it's finished with — whereas every future token attends
// back over every past key and value. That asymmetry is the whole trick.
type KVCache struct {
	cfg    GPTConfig
	layers []LayerKV
	n      int // positions stored
}

// LayerKV holds one layer's cached keys and values, indexed
// [kvHead][position][HeadDim].
//
// K is stored already QK-normed and rotated — the form attention consumes — so
// none of that work is repeated. V is raw, since nothing is applied to it.
type LayerKV struct {
	K, V [][][]float32
}

func NewKVCache(cfg GPTConfig) *KVCache {
	c := &KVCache{cfg: cfg, layers: make([]LayerKV, cfg.NLayer)}
	for i := range c.layers {
		c.layers[i] = *newLayerKV(cfg.NKVHead, 0)
	}
	return c
}

func newLayerKV(nKVHead, capacity int) *LayerKV {
	l := &LayerKV{
		K: make([][][]float32, nKVHead),
		V: make([][][]float32, nKVHead),
	}
	for i := range l.K {
		l.K[i] = make([][]float32, 0, capacity)
		l.V[i] = make([][]float32, 0, capacity)
	}
	return l
}

// Len is how many positions are cached — also the absolute position the next
// token will occupy, which is what the rotary tables get indexed by.
func (c *KVCache) Len() int { return c.n }

// Reset empties the cache while keeping the allocated backing arrays, so
// starting a new generation doesn't re-allocate everything.
func (c *KVCache) Reset() {
	for i := range c.layers {
		for h := range c.layers[i].K {
			c.layers[i].K[h] = c.layers[i].K[h][:0]
			c.layers[i].V[h] = c.layers[i].V[h][:0]
		}
	}
	c.n = 0
}

// BytesPerToken is what each additional position costs.
//
// For Qwen3-0.6B: 28 layers x 8 kv heads x 128 dims x 2 tensors x 4 bytes =
// 229KB per token. Over the full 40960-token context that's 9.4GB — and it
// would be 18.8GB without grouped-query attention, since NKVHead is half NHead.
func (c *KVCache) BytesPerToken() int {
	return c.cfg.NLayer * c.cfg.NKVHead * c.cfg.HeadDim * 2 * 4
}

// Bytes is the cache's current size, ignoring slice-header overhead.
func (c *KVCache) Bytes() int { return c.n * c.BytesPerToken() }

// compatibleWith rejects a cache built for a different model. Without this the
// mismatch shows up as an index panic several layers deep.
func (c *KVCache) compatibleWith(cfg GPTConfig) error {
	switch {
	case c.cfg.NLayer != cfg.NLayer:
		return fmt.Errorf("cache has %d layers, model has %d", c.cfg.NLayer, cfg.NLayer)
	case c.cfg.NKVHead != cfg.NKVHead:
		return fmt.Errorf("cache has %d kv heads, model has %d", c.cfg.NKVHead, cfg.NKVHead)
	case c.cfg.HeadDim != cfg.HeadDim:
		return fmt.Errorf("cache has head dim %d, model has %d", c.cfg.HeadDim, cfg.HeadDim)
	}
	return nil
}

// append stores one position's k and v for a layer's kv head.
func (l *LayerKV) append(kvHead int, k, v []float32) {
	l.K[kvHead] = append(l.K[kvHead], k)
	l.V[kvHead] = append(l.V[kvHead], v)
}
