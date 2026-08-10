package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
)

// Safetensors reads HuggingFace's .safetensors format, which is refreshingly
// simple: an 8-byte little-endian header length, that many bytes of JSON
// describing every tensor, then one flat blob of raw tensor bytes.
//
//	[8 bytes: uint64 headerLen][headerLen bytes: JSON][raw tensor data...]
//
// The JSON maps tensor name -> {dtype, shape, data_offsets}, where the offsets
// are relative to the start of the raw data section. There's no compression and
// no interleaving, so a tensor is just a subslice.
type Safetensors struct {
	tensors map[string]tensorEntry
	// data holds the raw bytes section for each shard the tensor came from.
	data map[string][]byte
}

type tensorEntry struct {
	DType   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`

	shard string // which file's data section Offsets index into
}

// maxHeaderLen guards against a corrupt or hostile length prefix asking us to
// allocate gigabytes before we've read a single tensor.
const maxHeaderLen = 100 << 20 // 100MB

// OpenSafetensors reads a single .safetensors file.
func OpenSafetensors(path string) (*Safetensors, error) {
	st := &Safetensors{
		tensors: make(map[string]tensorEntry),
		data:    make(map[string][]byte),
	}
	if err := st.addShard(path); err != nil {
		return nil, err
	}
	return st, nil
}

// OpenSafetensorsDir opens a checkpoint directory. Small models ship a single
// model.safetensors; larger ones shard across model-00001-of-0000N.safetensors
// with a model.safetensors.index.json naming which tensor lives where.
func OpenSafetensorsDir(dir string) (*Safetensors, error) {
	st := &Safetensors{
		tensors: make(map[string]tensorEntry),
		data:    make(map[string][]byte),
	}

	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if raw, err := os.ReadFile(indexPath); err == nil {
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &index); err != nil {
			return nil, fmt.Errorf("parsing %q: %w", indexPath, err)
		}
		// Load each distinct shard once, not once per tensor that lives in it.
		shards := make(map[string]bool, len(index.WeightMap))
		for _, file := range index.WeightMap {
			shards[file] = true
		}
		names := make([]string, 0, len(shards))
		for file := range shards {
			names = append(names, file)
		}
		sort.Strings(names) // deterministic load order
		for _, file := range names {
			if err := st.addShard(filepath.Join(dir, file)); err != nil {
				return nil, err
			}
		}
		return st, nil
	}

	single := filepath.Join(dir, "model.safetensors")
	if err := st.addShard(single); err != nil {
		return nil, fmt.Errorf("no model.safetensors.index.json and %w", err)
	}
	return st, nil
}

func (st *Safetensors) addShard(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("reading %q: %w", path, err)
	}
	if len(raw) < 8 {
		return fmt.Errorf("%q is too short to be safetensors", path)
	}

	headerLen := binary.LittleEndian.Uint64(raw[:8])
	if headerLen > maxHeaderLen || 8+headerLen > uint64(len(raw)) {
		return fmt.Errorf("%q has an implausible header length %d", path, headerLen)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+headerLen], &header); err != nil {
		return fmt.Errorf("parsing safetensors header in %q: %w", path, err)
	}

	body := raw[8+headerLen:]
	st.data[path] = body

	for name, rawEntry := range header {
		// __metadata__ is free-form provenance, not a tensor.
		if name == "__metadata__" {
			continue
		}
		var e tensorEntry
		if err := json.Unmarshal(rawEntry, &e); err != nil {
			return fmt.Errorf("parsing entry %q in %q: %w", name, path, err)
		}
		lo, hi := e.Offsets[0], e.Offsets[1]
		if lo < 0 || hi < lo || hi > int64(len(body)) {
			return fmt.Errorf("tensor %q in %q has out-of-range offsets [%d,%d)", name, path, lo, hi)
		}
		e.shard = path
		st.tensors[name] = e
	}
	return nil
}

// Names returns every tensor name, sorted.
func (st *Safetensors) Names() []string {
	out := make([]string, 0, len(st.tensors))
	for name := range st.tensors {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Has reports whether a tensor is present. Used to detect tied embeddings,
// where lm_head.weight simply doesn't exist in the checkpoint.
func (st *Safetensors) Has(name string) bool {
	_, ok := st.tensors[name]
	return ok
}

// Shape returns a tensor's dimensions.
func (st *Safetensors) Shape(name string) ([]int, error) {
	e, ok := st.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in checkpoint", name)
	}
	return e.Shape, nil
}

// Tensor decodes a tensor to float32, whatever it was stored as. Shapes are
// row-major and flattened — a (out, in) weight comes back as out*in floats,
// which is exactly the layout Linear.Weight wants, so no transpose is needed.
func (st *Safetensors) Tensor(name string) ([]float32, error) {
	e, ok := st.tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %q not found in checkpoint", name)
	}
	buf := st.data[e.shard][e.Offsets[0]:e.Offsets[1]]

	n := 1
	for _, d := range e.Shape {
		n *= d
	}

	width, ok := dtypeWidth(e.DType)
	if !ok {
		return nil, fmt.Errorf("tensor %q has unsupported dtype %q", name, e.DType)
	}
	if len(buf) != n*width {
		return nil, fmt.Errorf("tensor %q: shape %v implies %d bytes of %s but the slice is %d",
			name, e.Shape, n*width, e.DType, len(buf))
	}

	out := make([]float32, n)
	switch e.DType {
	case "F32":
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case "BF16":
		// bfloat16 is just float32 with the low 16 mantissa bits chopped off,
		// so widening is a shift — no exponent rebiasing, no special cases.
		for i := range out {
			out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[i*2:])) << 16)
		}
	case "F16":
		// IEEE half has a 5-bit exponent against float32's 8, so this one does
		// need real rebiasing plus subnormal handling.
		for i := range out {
			out[i] = f16to32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	}
	return out, nil
}

func dtypeWidth(dtype string) (int, bool) {
	switch dtype {
	case "F32":
		return 4, true
	case "BF16", "F16":
		return 2, true
	}
	return 0, false
}

// f16to32 widens an IEEE 754 half to a float32.
func f16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// Subnormal half, but a normal float32: shift the mantissa left until
		// the implicit leading 1 appears, decrementing the exponent to match.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13) // ±Inf / NaN
	default:
		return math.Float32frombits(sign | (exp-15+127)<<23 | mant<<13)
	}
}
