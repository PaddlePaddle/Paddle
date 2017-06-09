package pserver_test

import (
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/pserver"
)

func TestFull(t *testing.T) {
	s := pserver.NewService()
	var p pserver.Parameter
	p.Name = "param_a"
	p.Content = []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}
	p.ElementType = pserver.Int32
	err := s.InitParam(pserver.ParameterWithConfig{p, nil}, nil)
	if err != nil {
		t.FailNow()
	}

	var p1 pserver.Parameter
	p1.Name = "param_b"
	p1.Content = []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	p1.ElementType = pserver.Float32
	err = s.InitParam(pserver.ParameterWithConfig{p1, nil}, nil)
	if err != nil {
		t.FailNow()
	}

	err = s.FinishInitParams(0, nil)
	if err != nil {
		t.FailNow()
	}

	var param pserver.Parameter
	err = s.GetParam("param_b", &param)
	if err != nil {
		t.FailNow()
	}

	if !reflect.DeepEqual(param, p1) {
		t.FailNow()
	}

	g1, g2 := pserver.Gradient(p1), pserver.Gradient(p)
	err = s.SendGrad(g1, nil)
	if err != nil {
		t.FailNow()
	}
	err = s.SendGrad(g2, nil)

	if err != nil {
		t.FailNow()
	}

	var param1 pserver.Parameter
	err = s.GetParam("param_a", &param1)
	if err != nil {
		t.FailNow()
	}

	// don't compare content, since it's already changed by
	// gradient update.
	param1.Content = nil
	p.Content = nil

	if !reflect.DeepEqual(param1, p) {
		t.FailNow()
	}
}

func TestMultipleInit(t *testing.T) {
	s := pserver.NewService()
	err := s.FinishInitParams(0, nil)
	if err != nil {
		t.FailNow()
	}

	err = s.FinishInitParams(0, nil)
	if err != pserver.ErrAlreadyInitialized {
		t.FailNow()
	}
}

func TestUninitialized(t *testing.T) {
	s := pserver.NewService()
	err := s.SendGrad(pserver.Gradient{}, nil)
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
		var param pserver.Parameter
		err := s.GetParam("param_a", &param)
		if err != nil {
			t.FailNow()
		}
		wg.Done()
		ch <- struct{}{}
	}()

	wg.Add(1)
	go func() {
		err := s.Save("", nil)
		if err != nil {
			t.FailNow()
		}
		wg.Done()
		ch <- struct{}{}
	}()

	time.Sleep(50 * time.Millisecond)

	select {
	case <-ch:
		// some function returned before initialization is completed.
		t.FailNow()
	default:
	}

	var p pserver.Parameter
	p.Name = "param_a"
	p.Content = []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}
	p.ElementType = pserver.Int32
	err := s.InitParam(pserver.ParameterWithConfig{p, nil}, nil)
	if err != nil {
		t.FailNow()
	}

	err = s.FinishInitParams(0, nil)
	if err != nil {
		t.FailNow()
	}

	wg.Wait()
}
