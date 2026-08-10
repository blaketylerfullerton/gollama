package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// --- test helpers for building safetensors files by hand ---------------------

type stTensor struct {
	name  string
	dtype string
	shape []int
	data  []byte
}

// writeSafetensors emits the real on-disk format: 8-byte LE header length,
// JSON header, then the concatenated raw tensor bytes.
func writeSafetensors(t *testing.T, path string, tensors []stTensor) {
	t.Helper()

	header := make(map[string]any, len(tensors))
	var body []byte
	for _, tn := range tensors {
		lo := len(body)
		body = append(body, tn.data...)
		header[tn.name] = map[string]any{
			"dtype":        tn.dtype,
			"shape":        tn.shape,
			"data_offsets": []int{lo, len(body)},
		}
	}
	header["__metadata__"] = map[string]string{"format": "pt"}

	hj, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshaling header: %v", err)
	}

	buf := make([]byte, 8, 8+len(hj)+len(body))
	binary.LittleEndian.PutUint64(buf, uint64(len(hj)))
	buf = append(buf, hj...)
	buf = append(buf, body...)

	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("writing %q: %v", path, err)
	}
}

func f32Bytes(vals ...float32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func u16Bytes(vals ...uint16) []byte {
	buf := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	return buf
}

// --- dtype decoding ---------------------------------------------------------

func TestTensorF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	want := []float32{0, 1, -2.5, 3.75}
	writeSafetensors(t, path, []stTensor{
		{"w", "F32", []int{2, 2}, f32Bytes(want...)},
	})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := st.Tensor("w")
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("index %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestTensorBF16(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")

	// bfloat16 keeps float32's exponent, so these are exact once widened.
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x3F80, 1},
		{0xC000, -2},
		{0x3F00, 0.5},
		{0xBF80, -1},
	}
	bits := make([]uint16, len(cases))
	for i, c := range cases {
		bits[i] = c.bits
	}
	writeSafetensors(t, path, []stTensor{
		{"w", "BF16", []int{len(cases)}, u16Bytes(bits...)},
	})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := st.Tensor("w")
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		if got[i] != c.want {
			t.Errorf("bits %#04x: got %v, want %v", c.bits, got[i], c.want)
		}
	}
}

func TestTensorF16(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")

	cases := []struct {
		bits uint16
		want float32
		desc string
	}{
		{0x0000, 0, "zero"},
		{0x8000, 0, "negative zero"},
		{0x3C00, 1, "one"},
		{0x4000, 2, "two"},
		{0xC000, -2, "negative two"},
		{0x3800, 0.5, "half"},
		{0x7BFF, 65504, "largest normal"},
		{0x0001, math.Float32frombits(0x33800000), "smallest subnormal 2^-24"},
		{0x03FF, 6.0975552e-05, "largest subnormal"},
	}
	bits := make([]uint16, len(cases))
	for i, c := range cases {
		bits[i] = c.bits
	}
	writeSafetensors(t, path, []stTensor{
		{"w", "F16", []int{len(cases)}, u16Bytes(bits...)},
	})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := st.Tensor("w")
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range cases {
		if got[i] != c.want {
			t.Errorf("%s (bits %#04x): got %v, want %v", c.desc, c.bits, got[i], c.want)
		}
	}

	// Infinity is a separate check since it can't go in the equality table above.
	writeSafetensors(t, path, []stTensor{
		{"inf", "F16", []int{2}, u16Bytes(0x7C00, 0xFC00)},
	})
	st, err = OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err = st.Tensor("inf")
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(float64(got[0]), 1) || !math.IsInf(float64(got[1]), -1) {
		t.Errorf("got %v, want [+Inf -Inf]", got)
	}
}

// --- header and metadata ----------------------------------------------------

func TestNamesSkipsMetadata(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	writeSafetensors(t, path, []stTensor{
		{"b", "F32", []int{1}, f32Bytes(1)},
		{"a", "F32", []int{1}, f32Bytes(2)},
	})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	names := st.Names()
	if len(names) != 2 || names[0] != "a" || names[1] != "b" {
		t.Errorf("got %v, want sorted [a b] with __metadata__ excluded", names)
	}
	if st.Has("__metadata__") {
		t.Error("__metadata__ should not be exposed as a tensor")
	}
}

func TestShapeAndMissingTensor(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	writeSafetensors(t, path, []stTensor{
		{"w", "F32", []int{3, 2}, f32Bytes(1, 2, 3, 4, 5, 6)},
	})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	shape, err := st.Shape("w")
	if err != nil {
		t.Fatal(err)
	}
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 2 {
		t.Errorf("got shape %v, want [3 2]", shape)
	}
	if _, err := st.Tensor("nope"); err == nil {
		t.Error("expected an error for a missing tensor")
	}
}

func TestRejectsBadHeaderLength(t *testing.T) {
	dir := t.TempDir()

	tooShort := filepath.Join(dir, "short.safetensors")
	if err := os.WriteFile(tooShort, []byte{1, 2, 3}, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenSafetensors(tooShort); err == nil {
		t.Error("expected an error for a file shorter than the length prefix")
	}

	// A length prefix that overruns the file must fail rather than panic.
	overrun := filepath.Join(dir, "overrun.safetensors")
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint64(buf, 1<<40)
	if err := os.WriteFile(overrun, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenSafetensors(overrun); err == nil {
		t.Error("expected an error for an implausible header length")
	}
}

func TestUnsupportedDtype(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	writeSafetensors(t, path, []stTensor{
		{"w", "I64", []int{1}, make([]byte, 8)},
	})

	st, err := OpenSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := st.Tensor("w"); err == nil {
		t.Error("expected an error for an unsupported dtype")
	}
}

// --- sharded checkpoints ----------------------------------------------------

func TestOpenSafetensorsDirSharded(t *testing.T) {
	dir := t.TempDir()
	writeSafetensors(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), []stTensor{
		{"first", "F32", []int{2}, f32Bytes(1, 2)},
	})
	writeSafetensors(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), []stTensor{
		{"second", "F32", []int{2}, f32Bytes(3, 4)},
	})
	index := `{"weight_map":{
		"first":"model-00001-of-00002.safetensors",
		"second":"model-00002-of-00002.safetensors"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}

	st, err := OpenSafetensorsDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	// Offsets are shard-relative, so "second" resolving to [3 4] rather than
	// [1 2] is what proves each tensor indexes into its own shard's bytes.
	got, err := st.Tensor("second")
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 3 || got[1] != 4 {
		t.Errorf("got %v, want [3 4] — shard offsets may be crossing files", got)
	}
}

func TestOpenSafetensorsDirSingleFile(t *testing.T) {
	dir := t.TempDir()
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), []stTensor{
		{"w", "F32", []int{1}, f32Bytes(7)},
	})

	st, err := OpenSafetensorsDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := st.Tensor("w")
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 7 {
		t.Errorf("got %v, want [7]", got)
	}
}
