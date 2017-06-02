package recordio

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestChunkHead(t *testing.T) {
	assert := assert.New(t)

	c := &Header{
		checkSum:       123,
		compressor:     456,
		compressedSize: 789,
	}

	var buf bytes.Buffer
	_, e := c.write(&buf)
	assert.Nil(e)

	cc, e := parseHeader(&buf)
	assert.Nil(e)
	assert.Equal(c, cc)
}

func TestWriteAndRead(t *testing.T) {
	assert := assert.New(t)

	data := []string{
		"12345",
		"1234",
		"12"}

	var buf bytes.Buffer
	w := NewWriter(&buf, 10, NoCompression) // use a small maxChunkSize.

	n, e := w.Write([]byte(data[0])) // not exceed chunk size.
	assert.Nil(e)
	assert.Equal(5, n)

	n, e = w.Write([]byte(data[1])) // not exceed chunk size.
	assert.Nil(e)
	assert.Equal(4, n)

	n, e = w.Write([]byte(data[2])) // exeeds chunk size, dump and create a new chunk.
	assert.Nil(e)
	assert.Equal(n, 2)

	assert.Nil(w.Close()) // flush the second chunk.
	assert.Nil(w.Writer)

	n, e = w.Write([]byte("anything")) // not effective after close.
	assert.NotNil(e)
	assert.Equal(n, 0)

	idx, e := LoadIndex(bytes.NewReader(buf.Bytes()))
	assert.Nil(e)
	assert.Equal([]uint32{2, 1}, idx.chunkLens)
	assert.Equal(
		[]int64{0,
			int64(4 + // magic number
				unsafe.Sizeof(Header{}) +
				5 + // first record
				4 + // second record
				2*4)}, // two record legnths
		idx.chunkOffsets)

	s := NewRangeScanner(bytes.NewReader(buf.Bytes()), idx, -1, -1)
	i := 0
	for s.Scan() {
		assert.Equal(data[i], string(s.Record()))
		i++
	}
}

func TestWriteEmptyFile(t *testing.T) {
	assert := assert.New(t)

	var buf bytes.Buffer
	w := NewWriter(&buf, 10, NoCompression) // use a small maxChunkSize.
	assert.Nil(w.Close())
	assert.Equal(0, buf.Len())

	idx, e := LoadIndex(bytes.NewReader(buf.Bytes()))
	assert.Nil(e)
	assert.Equal(0, idx.NumRecords())
}
