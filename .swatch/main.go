package main

import (
	"fmt"
	"math"
	"strconv"

	"github.com/blaketylerfullerton/GoLlama/tools/amber"
)

func lum(hex string) float64 {
	c := func(i int) float64 {
		v, _ := strconv.ParseInt(hex[i:i+2], 16, 0)
		s := float64(v) / 255
		if s <= 0.04045 {
			return s / 12.92
		}
		return math.Pow((s+0.055)/1.055, 2.4)
	}
	return 0.2126*c(1) + 0.7152*c(3) + 0.0722*c(5)
}
func ratio(hex string) float64 { return (lum(hex) + 0.05) / 0.05 }

func main() {
	fmt.Println("lvl  data(Ramp)      ratio   neutral        ratio")
	for i := 0; i < 10; i++ {
		d, n := string(amber.Ramp[i]), string(amber.Neutral[i])
		fmt.Printf("%2d   %s  %5.2f:1   %s  %5.2f:1\n", i, d, ratio(d), n, ratio(n))
	}
	fmt.Printf("\nAccent  = %s  %.2f:1\n", amber.At(amber.Accent), ratio(string(amber.At(amber.Accent))))
	fmt.Printf("Alert   = %s  %.2f:1\n", amber.Alert, ratio(string(amber.Alert)))
	fmt.Println("\ntext-bearing levels must clear 4.5:1 on black:")
	for _, t := range []struct {
		n string
		l int
	}{{"Muted", amber.Muted}, {"Body", amber.Body}, {"Strong", amber.Strong}} {
		r := ratio(string(amber.N(t.l)))
		ok := "PASS"
		if r < 4.5 {
			ok = "FAIL"
		}
		fmt.Printf("  %-7s %s %5.2f:1  %s\n", t.n, amber.N(t.l), r, ok)
	}
	fmt.Println("\nhue separation (accent vs alert):")
	showHues([][2]string{{"Accent", string(amber.At(amber.Accent))}, {"Alert", string(amber.Alert)}})
	r := ratio(string(amber.At(amber.Accent)))
	ok := "PASS"
	if r < 4.5 {
		ok = "FAIL"
	}
	fmt.Printf("  %-7s %s %5.2f:1  %s\n", "Accent", amber.At(amber.Accent), r, ok)
}
