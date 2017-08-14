// Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pserver

import (
	"io/ioutil"
	"testing"
)

func TestOptimizerCreateRelease(t *testing.T) {
	p := Parameter{
		Name:        "a",
		ElementType: Int32,
	}
	p.Content = []byte{1, 3}
	config, err := ioutil.ReadFile("./client/c/test/testdata/optimizer.pb")
	if err != nil {
		t.Fatalf("read optimizer proto failed")
	}
	param := ParameterWithConfig{
		Param:  p,
		Config: config,
	}
	o := newOptimizer(param, nil)
	o.Cleanup()
}
