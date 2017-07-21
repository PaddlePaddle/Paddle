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
