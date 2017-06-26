package pserver

import (
	"io/ioutil"
	"testing"
)

func TestOptimizerCreateRelease(t *testing.T) {
	p := Parameter{
		Name:        "a",
		ElementType: Float32,
	}
	p.Content = []byte{0.1, 0.3}
	config, err := ioutil.ReadFile("./cclient/test/testdata/optimizer.pb.txt")

	param := ParameterWithConfig{
		Param:  p,
		Config: config,
	}
	o := newOptimizer(param)
	o.Cleanup()
}
