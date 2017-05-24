package master

import "testing"

func TestPartitionCount(t *testing.T) {
	cs := make([]Chunk, 100)
	ts := partition(cs, 20)
	if len(ts) != 20 {
		t.Error(len(ts))
	}

	cs = make([]Chunk, 101)
	ts = partition(cs, 20)
	if len(ts) != 21 {
		t.Error(len(ts))
	}

	ts = partition(cs, 200)
	if len(ts) != 101 {
		t.Error(len(ts))
	}
}

func TestPartionIndex(t *testing.T) {
	cs := make([]Chunk, 100)
	ts := partition(cs, 20)
	for i := range ts {
		if ts[i].Task.ID != i {
			t.Error(ts[i], i)
		}
	}
}
