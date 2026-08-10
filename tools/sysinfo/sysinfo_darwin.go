//go:build darwin

package sysinfo

import (
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// detect reads the machine description out of sysctl.
//
// The syscall package can do sysctlbyname without shelling out, but the string
// keys we want (machdep.cpu.brand_string, hw.perflevel*) then need a cgo-free
// wrapper each, and the binary here already assumes a developer machine. One
// exec of sysctl with every key at once costs a few milliseconds.
func detect(in *Info) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	keys := []string{
		"machdep.cpu.brand_string",
		"hw.model",
		"hw.memsize",
		"hw.perflevel0.logicalcpu", // performance cores; absent on Intel
		"hw.perflevel1.logicalcpu", // efficiency cores
	}
	// -n prints values only, one per line, in the order asked. A key that
	// doesn't exist is skipped with a message on stderr, which would shift every
	// later line — so ask for them one per invocation-safe pair instead.
	vals := make(map[string]string, len(keys))
	for _, k := range keys {
		if v, ok := sysctl(ctx, k); ok {
			vals[k] = v
		}
	}

	in.CPU = vals["machdep.cpu.brand_string"]
	in.Model = vals["hw.model"]
	if n, err := strconv.ParseUint(vals["hw.memsize"], 10, 64); err == nil {
		in.MemoryBytes = n
	}
	in.PCores, _ = strconv.Atoi(vals["hw.perflevel0.logicalcpu"])
	in.ECores, _ = strconv.Atoi(vals["hw.perflevel1.logicalcpu"])
	// On Intel Macs perflevel0 is every core and there is no perflevel1; a split
	// with nothing on the other side isn't a split worth showing.
	if in.ECores == 0 {
		in.PCores = 0
	}

	if name, ok := run(ctx, "scutil", "--get", "ComputerName"); ok {
		in.Host = name
	}
	in.GPU = gpu(ctx)
	in.AvailableBytes = available(ctx)
}

var pageSize = regexp.MustCompile(`page size of (\d+) bytes`)

// available approximates what macOS would hand out before it started
// compressing or swapping.
//
// There is no single "available" counter on Darwin the way Linux publishes
// MemAvailable, so this sums the page classes the kernel will reclaim without
// paging anything out: free pages, the inactive list, speculative read-ahead,
// and purgeable caches. Wired, active and already-compressed pages are excluded
// — those are the ones a new 2GB allocation would actually have to fight for.
// It's an estimate, and it's labelled as one on screen.
func available(ctx context.Context) uint64 {
	out, ok := run(ctx, "vm_stat")
	if !ok {
		return 0
	}
	m := pageSize.FindStringSubmatch(out)
	if m == nil {
		return 0
	}
	size, err := strconv.ParseUint(m[1], 10, 64)
	if err != nil {
		return 0
	}

	var pages uint64
	for _, line := range strings.Split(out, "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		switch strings.TrimSpace(key) {
		case "Pages free", "Pages inactive", "Pages speculative", "Pages purgeable":
		default:
			continue
		}
		// vm_stat writes counts with a trailing period: "Pages free: 12345."
		n, err := strconv.ParseUint(strings.Trim(strings.TrimSpace(value), "."), 10, 64)
		if err != nil {
			continue
		}
		pages += n
	}
	return pages * size
}

func sysctl(ctx context.Context, key string) (string, bool) {
	return run(ctx, "sysctl", "-n", key)
}

var gpuCores = regexp.MustCompile(`"gpu-core-count"\s*=\s*(\d+)`)

// gpu reports the integrated GPU's core count, which on Apple silicon is the
// number people quote when comparing chips. We don't use the GPU yet — every
// matmul in model/ is scalar Go on the CPU — so this is context for how much
// headroom is being left on the table, not a capability claim.
func gpu(ctx context.Context) string {
	out, ok := run(ctx, "ioreg", "-r", "-d", "1", "-k", "gpu-core-count")
	if !ok {
		return ""
	}
	m := gpuCores.FindStringSubmatch(out)
	if m == nil {
		return ""
	}
	return fmt.Sprintf("%s-core GPU (unused)", m[1])
}

func run(ctx context.Context, name string, args ...string) (string, bool) {
	out, err := exec.CommandContext(ctx, name, args...).Output()
	if err != nil {
		return "", false
	}
	s := strings.TrimSpace(string(out))
	return s, s != ""
}
