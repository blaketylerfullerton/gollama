// Package sysinfo reports what hardware this process is running on.
//
// It exists because the two numbers that decide how a local model feels are
// hardware numbers: how many cores can be thrown at the matmuls, and whether the
// weights fit in memory next to everything else already there. Printing them on
// the way in means the speed you get later isn't a mystery.
//
// Detection is deliberately cheap — sysctls and /proc reads, nothing that shells
// out to a profiler — because it runs before the first frame is drawn. Anything
// that can't be determined is left zero, and callers are expected to fall back
// rather than to treat a missing field as an error.
package sysinfo

import (
	"fmt"
	"os"
	"runtime"
)

// Info is a snapshot of the machine. Zero values mean "not reported on this
// platform", which is normal: only Apple silicon splits its cores into
// performance and efficiency, and only some machines name their GPU.
type Info struct {
	Host   string // computer name, e.g. "Blake's MacBook Air"
	CPU    string // marketing name, e.g. "Apple M4"
	Model  string // hardware identifier, e.g. "Mac16,13"
	Cores  int    // logical cores, always set
	PCores int    // performance cores, Apple silicon only
	ECores int    // efficiency cores, Apple silicon only
	GPU    string // e.g. "10-core GPU", empty when unknown

	MemoryBytes uint64 // total physical RAM, 0 when unknown

	OS         string
	Arch       string
	GoVersion  string
	GOMAXPROCS int // how many threads Go will actually run in parallel
}

// Detect fills in everything it can. It never fails: on an unsupported platform
// you still get the runtime fields, which are enough to render a screen.
func Detect() Info {
	in := Info{
		Cores:      runtime.NumCPU(),
		OS:         runtime.GOOS,
		Arch:       runtime.GOARCH,
		GoVersion:  runtime.Version(),
		GOMAXPROCS: runtime.GOMAXPROCS(0),
	}
	detect(&in) // platform-specific, best effort
	if in.Host == "" {
		if h, err := os.Hostname(); err == nil {
			in.Host = h
		}
	}
	if in.CPU == "" {
		in.CPU = runtime.GOARCH + " cpu"
	}
	return in
}

// CoreSummary describes the core count, mentioning the performance/efficiency
// split when the platform reports one. That split matters: on an M-series chip
// the efficiency cores are perhaps a third the throughput of a performance core,
// so "10 cores" oversells what a matmul actually gets.
func (in Info) CoreSummary() string {
	if in.PCores > 0 && in.ECores > 0 {
		return fmt.Sprintf("%d cores (%dP + %dE)", in.Cores, in.PCores, in.ECores)
	}
	return fmt.Sprintf("%d cores", in.Cores)
}

// Memory is the total RAM, or "unknown" when the platform didn't say.
func (in Info) Memory() string {
	if in.MemoryBytes == 0 {
		return "unknown"
	}
	return Bytes(int64(in.MemoryBytes))
}

// Platform is the GOOS/GOARCH pair the binary was built for.
func (in Info) Platform() string { return in.OS + "/" + in.Arch }

// Bytes formats a byte count the way a spec sheet would: 1.4 GB, 604 MB.
// Powers of 1024 with SI-looking suffixes, which is the convention every other
// model runner prints, so the numbers are comparable to what people expect.
func Bytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for m := n / unit; m >= unit && exp < 3; m /= unit {
		div *= unit
		exp++
	}
	v := float64(n) / float64(div)
	suffix := [...]string{"KB", "MB", "GB", "TB"}[exp]
	if v >= 100 {
		return fmt.Sprintf("%.0f %s", v, suffix)
	}
	return fmt.Sprintf("%.1f %s", v, suffix)
}
