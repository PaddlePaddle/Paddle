package recordio_test

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/PaddlePaddle/Paddle/go/recordio"
)

func TestWriteRead(t *testing.T) {
	const total = 1000
	var buf bytes.Buffer
	w := recordio.NewWriter(&buf, 0, -1)
	for i := 0; i < total; i++ {
		_, err := w.Write(make([]byte, i))
		if err != nil {
			t.Fatal(err)
		}
	}
	w.Close()

	idx, err := recordio.LoadIndex(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	if idx.NumRecords() != total {
		t.Fatal("num record does not match:", idx.NumRecords(), total)
	}

	s := recordio.NewRangeScanner(bytes.NewReader(buf.Bytes()), idx, -1, -1)
	i := 0
	for s.Scan() {
		if !reflect.DeepEqual(s.Record(), make([]byte, i)) {
			t.Fatal("not equal:", len(s.Record()), len(make([]byte, i)))
		}
		i++
	}

	if i != total {
		t.Fatal("total count not match:", i, total)
	}
}

func TestChunkIndex(t *testing.T) {
	const total = 1000
	var buf bytes.Buffer
	w := recordio.NewWriter(&buf, 0, -1)
	for i := 0; i < total; i++ {
		_, err := w.Write(make([]byte, i))
		if err != nil {
			t.Fatal(err)
		}
	}
	w.Close()

	idx, err := recordio.LoadIndex(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	if idx.NumChunks() != total {
		t.Fatal("unexpected chunk num:", idx.NumChunks(), total)
	}

	for i := 0; i < total; i++ {
		newIdx := idx.ChunkIndex(i)
		s := recordio.NewRangeScanner(bytes.NewReader(buf.Bytes()), newIdx, -1, -1)
		j := 0
		for s.Scan() {
			if !reflect.DeepEqual(s.Record(), make([]byte, i)) {
				t.Fatal("not equal:", len(s.Record()), len(make([]byte, i)))
			}
			j++
		}
		if j != 1 {
			t.Fatal("unexpected record per chunk:", j)
		}
	}
}
