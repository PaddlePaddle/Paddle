package pserver

import "testing"

func TestSGDCreateRelease(t *testing.T) {
	o := newOptimizer(sgd, 1)
	o.Cleanup()
}
