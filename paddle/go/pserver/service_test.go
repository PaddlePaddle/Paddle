package pserver_test

import (
	"reflect"
	"sync"
	"testing"

	"github.com/PaddlePaddle/Paddle/paddle/go/pserver"
)

func TestFull(t *testing.T) {
	s := pserver.NewService()
	var dummy int
	err := s.BeginInitParams(nil, &dummy)
	if err != nil {
		t.FailNow()
	}

	var p pserver.Parameter
	p.Name = "param_a"
	p.Content = []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}
	p.ElementType = pserver.Int32
	err = s.InitParam(pserver.ParameterWithConfig{p, nil}, &dummy)
	if err != nil {
		t.FailNow()
	}

	var p1 pserver.Parameter
	p1.Name = "param_b"
	p1.Content = []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	p1.ElementType = pserver.Float32
	err = s.InitParam(pserver.ParameterWithConfig{p1, nil}, &dummy)
	if err != nil {
		t.FailNow()
	}

	err = s.FinishInitParams(0, &dummy)
	if err != nil {
		t.FailNow()
	}

	var params []pserver.Parameter
	err = s.GetParams([]string{"param_b", "param_a"}, &params)
	if err != nil {
		t.FailNow()
	}

	if len(params) != 2 || !reflect.DeepEqual(params[0], p1) || !reflect.DeepEqual(params[0], p1) {
		t.FailNow()
	}

	grads := []pserver.Gradient{pserver.Gradient(p1), pserver.Gradient(p)}
	err = s.SendGrads(grads, &dummy)
	if err != nil {
		t.FailNow()
	}

	var params1 []pserver.Parameter
	err = s.GetParams([]string{"param_b", "param_a"}, &params1)
	if err != nil {
		t.FailNow()
	}

	if len(params) != 2 {
		t.FailNow()
	}

	// don't compare content, since it's already changed by
	// gradient update.
	params1[0].Content = nil
	params1[0].Content = nil
	p.Content = nil
	p1.Content = nil

	if !reflect.DeepEqual(params1[0], p1) || !reflect.DeepEqual(params1[0], p1) {
		t.FailNow()
	}
}

func TestMultipleInit(t *testing.T) {
	s := pserver.NewService()
	var dummy int
	err := s.BeginInitParams(nil, &dummy)
	if err != nil {
		t.FailNow()
	}

	// this is fine, it's possible for client to call init
	// multiple times.
	err = s.BeginInitParams(nil, &dummy)
	if err != nil {
		t.FailNow()
	}

	err = s.FinishInitParams(0, &dummy)
	if err != nil {
		t.FailNow()
	}

	err = s.FinishInitParams(0, &dummy)
	if err != pserver.ErrAlreadyInitialized {
		t.FailNow()
	}

	err = s.BeginInitParams(nil, &dummy)
	if err != pserver.ErrAlreadyInitialized {
		t.FailNow()
	}
}

func TestUninitialized(t *testing.T) {
	s := pserver.NewService()
	var dummy int
	err := s.SendGrads(nil, &dummy)
	if err != pserver.ErrUninitialized {
		t.FailNow()
	}
}

func TestBlockUntilInitialized(t *testing.T) {
	s := pserver.NewService()
	ch := make(chan struct{}, 2)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		var params []pserver.Parameter
		err := s.GetParams(nil, &params)
		if err != nil {
			t.FailNow()
		}
		wg.Done()
		ch <- struct{}{}
	}()

	wg.Add(1)
	go func() {
		var dummy int
		err := s.Save("", &dummy)
		if err != nil {
			t.FailNow()
		}
		wg.Done()
		ch <- struct{}{}
	}()

	var dummy int
	err := s.BeginInitParams(nil, &dummy)
	if err != nil {
		t.FailNow()
	}

	select {
	case <-ch:
		// some function returned before initialization is completed.
		t.FailNow()
	default:
	}

	err = s.FinishInitParams(0, &dummy)
	if err != nil {
		t.FailNow()
	}

	wg.Wait()
}
