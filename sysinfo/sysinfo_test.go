package sysinfo

import (
	"runtime"
	"strings"
	"testing"
)

// Detect has to work on a machine it knows nothing about: the welcome screen
// calls it before anything else and has no fallback if it panics or comes back
// empty.
func TestDetectAlwaysFillsRuntimeFields(t *testing.T) {
	in := Detect()
	if in.Cores < 1 {
		t.Errorf("Cores = %d, want at least 1", in.Cores)
	}
	if in.GOMAXPROCS < 1 {
		t.Errorf("GOMAXPROCS = %d, want at least 1", in.GOMAXPROCS)
	}
	if in.Platform() != runtime.GOOS+"/"+runtime.GOARCH {
		t.Errorf("Platform() = %q", in.Platform())
	}
	if in.CPU == "" {
		t.Error("CPU is empty; it should fall back to the architecture")
	}
	if in.Host == "" {
		t.Error("Host is empty; it should fall back to os.Hostname")
	}
}

// The P/E split is only shown when both halves are known — "10 cores (10P + 0E)"
// would be a worse answer than "10 cores".
func TestCoreSummary(t *testing.T) {
	for _, tc := range []struct {
		in   Info
		want string
	}{
		{Info{Cores: 10, PCores: 4, ECores: 6}, "10 cores (4P + 6E)"},
		{Info{Cores: 8}, "8 cores"},
		{Info{Cores: 8, PCores: 8}, "8 cores"},
	} {
		if got := tc.in.CoreSummary(); got != tc.want {
			t.Errorf("CoreSummary() = %q, want %q", got, tc.want)
		}
	}
}

func TestMemoryUnknown(t *testing.T) {
	if got := (Info{}).Memory(); got != "unknown" {
		t.Errorf("Memory() with no reading = %q, want %q", got, "unknown")
	}
}

func TestBytes(t *testing.T) {
	for _, tc := range []struct {
		n    int64
		want string
	}{
		{0, "0 B"},
		{999, "999 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{150 * 1024, "150 KB"},      // three digits drop the decimal
		{1_500_000_000, "1.4 GB"},   // a Qwen3-0.6B checkpoint
		{17_179_869_184, "16.0 GB"}, // 16 GB of RAM
		{4 * 1024 * 1024 * 1024 * 1024, "4.0 TB"},
	} {
		if got := Bytes(tc.n); got != tc.want {
			t.Errorf("Bytes(%d) = %q, want %q", tc.n, got, tc.want)
		}
	}
}

// Whatever the platform reports, it goes straight onto the screen, so a stray
// newline from a sysctl or a /proc read would break the layout.
func TestDetectedStringsAreSingleLine(t *testing.T) {
	in := Detect()
	for name, v := range map[string]string{
		"Host": in.Host, "CPU": in.CPU, "Model": in.Model, "GPU": in.GPU,
	} {
		if strings.ContainsAny(v, "\n\r\t") {
			t.Errorf("%s contains whitespace that would break a row: %q", name, v)
		}
	}
}
