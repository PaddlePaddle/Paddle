package pserver

import (
	"io/ioutil"
	"reflect"
	"testing"
)

func TestOptimizerCreateRelease(t *testing.T) {
	p := Parameter{
		Name:        "a",
		ElementType: Int32,
	}
	p.Content = []byte{1, 3}
	config, err := ioutil.ReadFile("./cclient/test/testdata/optimizer.pb.txt")
	if err != nil {
		t.Fatalf("read optimizer proto failed")
	}
	param := ParameterWithConfig{
		Param:  p,
		Config: config,
	}
	o := newOptimizer(param)
	o.Cleanup()
}

func TestOptimizerFull(t *testing.T) {
	p := Parameter{
		Name:        "a",
		ElementType: Float32,
	}
	p.Content = []byte{1, 3}
	config, err := ioutil.ReadFile("./cclient/test/testdata/optimizer.pb.txt")
	if err != nil {
		t.Fatalf("read optimizer proto failed")
	}
	param := ParameterWithConfig{
		Param:  p,
		Config: config,
	}
	o := newOptimizer(param)
	g := Gradient(p)
	if !reflect.DeepEqual(p.Content, o.GetWeights()) {
		t.FailNow()
	}
	o.UpdateParameter(g)
	o.Cleanup()
}
