package main

import "C"

//export GoAdder
func GoAdder(x, y int) int {
	return x + y
}

func main() {} // Required but ignored
