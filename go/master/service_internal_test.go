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

package master

import "testing"

func TestPartitionCount(t *testing.T) {
	cs := make([]Chunk, 100)
	ts := partition(cs, 5)
	if len(ts) != 20 {
		t.Error(len(ts))
	}

	cs = make([]Chunk, 101)
	ts = partition(cs, 5)
	if len(ts) != 21 {
		t.Error(len(ts))
	}

	ts = partition(cs, 1)
	if len(ts) != 101 {
		t.Error(len(ts))
	}

	ts = partition(cs, 0)
	if len(ts) != 101 {
		t.Error(len(ts))
	}
}

func TestPartionIndex(t *testing.T) {
	cs := make([]Chunk, 100)
	ts := partition(cs, 20)
	for i := range ts {
		// test auto increament ids
		if i > 0 && ts[i].Task.Meta.ID != ts[i-1].Task.Meta.ID+1 {
			t.Error(ts[i], i)
		}
	}
}
