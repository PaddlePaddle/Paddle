// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pserver_test

import (
	"fmt"
	"io/ioutil"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/PaddlePaddle/Paddle/go/pserver"
)

const (
	OptimizerConfig = "./client/c/test/testdata/optimizer.pb"
)

func TestServiceFull(t *testing.T) {
	var cp pserver.Checkpoint
	s, err := pserver.NewService(0, time.Hour, "", nil, cp)
	if err != nil {
		t.Error(err)
	}
	var p pserver.Parameter
	p.Name = "param_a"
	p.Content = []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}
	p.ElementType = pserver.Int32
	config, err := ioutil.ReadFile(OptimizerConfig)
	if err != nil {
		t.Fatalf("read optimizer proto failed")
	}

	err = s.InitParam(pserver.ParameterWithConfig{Param: p, Config: config}, nil)
	if err != nil {
		t.Fatal(err)
	}

	var p1 pserver.Parameter
	p1.Name = "param_b"
	p1.Content = []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	p1.ElementType = pserver.Float32
	err = s.InitParam(pserver.ParameterWithConfig{Param: p1, Config: config}, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = s.FinishInitParams(0, nil)
	if err != nil {
		t.Fatal(err)
	}

	var param pserver.Parameter
	err = s.GetParam("param_b", &param)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(param, p1) {
		t.Fatal("not equal:", param, p1)
	}

	g1, g2 := pserver.Gradient(p1), pserver.Gradient(p)

	err = s.SendGrad(g1, nil)
	if err != nil {
		t.Fatal(err)
	}
	err = s.SendGrad(g2, nil)

	if err != nil {
		t.Fatal(err)
	}

	var param1 pserver.Parameter
	err = s.GetParam("param_a", &param1)
	if err != nil {
		t.Fatal(err)
	}

	// don't compare content, since it's already changed by
	// gradient update.
	param1.Content = nil
	p.Content = nil

	if !reflect.DeepEqual(param1, p) {
		t.Fatal("not equal:", param1, p)
	}
}

func TestMultipleInit(t *testing.T) {
	var cp pserver.Checkpoint
	s, err := pserver.NewService(0, time.Hour, "", nil, cp)
	if err != nil {
		t.Fatal(err)
	}
	err = s.FinishInitParams(0, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = s.FinishInitParams(0, nil)
	if err.Error() != pserver.AlreadyInitialized {
		t.Fatal(err)
	}
}

func TestUninitialized(t *testing.T) {
	var cp pserver.Checkpoint
	s, err := pserver.NewService(0, time.Hour, "", nil, cp)
	err = s.SendGrad(pserver.Gradient{}, nil)
	if err.Error() != pserver.Uninitialized {
		t.Fatal(err)
	}
}

func TestBlockUntilInitialized(t *testing.T) {
	var cp pserver.Checkpoint
	s, err := pserver.NewService(0, time.Hour, "", nil, cp)
	if err != nil {
		t.Error(err)
	}
	ch := make(chan struct{}, 2)
	errCh := make(chan error, 2)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		var param pserver.Parameter
		err := s.GetParam("param_a", &param)
		if err != nil {
			errCh <- err
		}
		wg.Done()
		ch <- struct{}{}
	}()

	time.Sleep(50 * time.Millisecond)

	select {
	case <-ch:
		// some function returned before initialization is completed.
		t.FailNow()
	case <-errCh:
		t.FailNow()
	default:
	}

	var p pserver.Parameter
	p.Name = "param_a"
	p.Content = []byte{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}
	p.ElementType = pserver.Int32
	config, err := ioutil.ReadFile(OptimizerConfig)
	if err != nil {
		t.Fatalf("read optimizer proto failed")
	}
	err = s.InitParam(pserver.ParameterWithConfig{Param: p, Config: config}, nil)

	if err != nil {
		t.Fatal(err)
	}

	err = s.FinishInitParams(0, nil)
	if err != nil {
		t.Fatal(err)
	}

	wg.Wait()
}

func TestGradientString(t *testing.T) {
	g := pserver.Parameter{}
	g.ElementType = pserver.Float32
	g.Content = []byte{0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40, 0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40}
	if g.String() != "[3.3702806e+12 2.142699 3.3702806e+12 2.142699]" {
		t.Fatal("get float data error!")
	}

	g.Content = []byte{0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40,
		0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40}
	if g.String() != "[3.3702806e+12 2.142699 3.3702806e+12 2.142699 3.3702806e+12 2.142699 3.3702806e+12 2.142699 3.3702806e+12 2.142699...3.3702806e+12 2.142699 3.3702806e+12 2.142699 3.3702806e+12 2.142699 3.3702806e+12 2.142699 3.3702806e+12 2.142699]" {
		t.Fatal("get float data error!", g.String())
	}
	fmt.Println(g)
}
