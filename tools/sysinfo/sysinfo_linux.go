//go:build linux

package sysinfo

import (
	"os"
	"strconv"
	"strings"
)

// detect reads /proc, which is always there and always cheap. Nothing is shelled
// out: everything we want is a file.
func detect(in *Info) {
	in.CPU = firstField("/proc/cpuinfo", "model name")
	if in.CPU == "" {
		// arm64 kernels don't publish a model name; this is the closest thing.
		in.CPU = firstField("/proc/cpuinfo", "Hardware")
	}
	in.MemoryBytes = memKB("MemTotal")
	// MemAvailable is the kernel's own estimate of what a new allocation can
	// have without swapping — better than free+cached, which overstates it.
	in.AvailableBytes = memKB("MemAvailable")
	in.Model = strings.TrimSpace(read("/sys/devices/virtual/dmi/id/product_name"))
}

// firstField returns the value of the first "key : value" line matching key.
// /proc/cpuinfo repeats the block once per core and we only need one.
func firstField(path, key string) string {
	for _, line := range strings.Split(read(path), "\n") {
		name, value, ok := strings.Cut(line, ":")
		if ok && strings.TrimSpace(name) == key {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

// memKB reads one /proc/meminfo field, which is reported in kB.
func memKB(key string) uint64 {
	v := firstField("/proc/meminfo", key)
	kb, err := strconv.ParseUint(strings.TrimSuffix(v, " kB"), 10, 64)
	if err != nil {
		return 0
	}
	return kb * 1024
}

func read(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}
