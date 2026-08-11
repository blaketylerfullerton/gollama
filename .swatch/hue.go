package main

import (
	"fmt"
	"math"
	"strconv"
)

func hueOf(hex string) float64 {
	f := func(i int) float64 {
		v, _ := strconv.ParseInt(hex[i:i+2], 16, 0)
		return float64(v) / 255
	}
	r, g, b := f(1), f(3), f(5)
	mx, mn := math.Max(r, math.Max(g, b)), math.Min(r, math.Min(g, b))
	d := mx - mn
	if d == 0 {
		return 0
	}
	var h float64
	switch mx {
	case r:
		h = math.Mod((g-b)/d, 6)
	case g:
		h = (b-r)/d + 2
	default:
		h = (r-g)/d + 4
	}
	h *= 60
	if h < 0 {
		h += 360
	}
	return h
}

func showHues(pairs [][2]string) {
	for _, p := range pairs {
		fmt.Printf("  %-8s %s  hue %5.1f°  %5.2f:1\n", p[0], p[1], hueOf(p[1]), ratio(p[1]))
	}
}
