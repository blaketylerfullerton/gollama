//go:build !darwin && !linux

package sysinfo

// detect has nothing platform-specific to add here. Detect still fills in the
// runtime fields — core count, platform, Go version — so the screen renders
// with holes rather than not at all.
func detect(*Info) {}
